"""OpenAI / Anthropic / Responses-compatible server on deepseek-recipe (the same request
conversion, prompt rendering and output parsing as DeepSeek's API), backed by the local engine.

DeepInfer's public release is offline-only; this follows deepseek-recipe's server-py example with
the mock backend replaced by `LLM.stream`. One sequence runs at a time (a 24 GB Mac streams
experts for one), requests queue on a lock, and a follow-up chat turn reuses the previous turn's
state when its prompt extends it.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from time import time
from uuid import uuid4

from deepseek_recipe import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ConversionError,
    ConversionOptions,
    DeepseekV41Encoding,
    InferenceChunk,
    InferenceFinishReason,
    MessagesRequest,
    MessagesResponse,
    ModelVersion,
    PromptUsage,
    ResponsesRequest,
    ResponsesResponse,
    StreamProcessor,
    Tokenizer,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .engine import EOS_ID, LLM, SamplingParams, encode_conversation

PROTOCOLS = {
    "chat_completions": (ChatCompletionRequest, ChatCompletionResponse),
    "responses": (ResponsesRequest, ResponsesResponse),
    "messages": (MessagesRequest, MessagesResponse),
}
CONTEXT_LIMIT = 262144


def conversion_options() -> ConversionOptions:
    return ConversionOptions().with_model_alias("deepseek-flash", ModelVersion.DeepseekV41Flash).with_model_alias(
        "deepseek-v4-flash", ModelVersion.DeepseekV41Flash
    )


def sampling_from(conv, n_prompt: int) -> SamplingParams:
    o = conv.inference_options
    sp = SamplingParams(
        temperature=o.temperature if o.temperature is not None else 1.0,
        top_p=o.top_p if o.top_p is not None else 1.0,
        max_tokens=conv.resolve_max_tokens(n_prompt, CONTEXT_LIMIT),
        seed=o.seed,
    )
    c = o.output_constraint
    if c is not None and c.kind in ("json", "regex"):
        sp.so_args = {"schema_type": c.kind, "schema": c.schema}
    return sp


def create_app(llm: LLM) -> FastAPI:
    app = FastAPI(title="DeepInfer-Mac", version="0.1.0")
    tok = Tokenizer.from_file(str(llm.root / "tokenizer.json"))
    gate = threading.Lock()

    @app.exception_handler(ConversionError)
    async def conversion_error(_req: Request, exc: ConversionError) -> Response:
        return Response(content=exc.body, status_code=exc.status_code, media_type="application/json")

    async def infer(prompt_ids: list[int], sp: SamplingParams, images=None) -> AsyncIterator[InferenceChunk]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def run():
            with gate:
                cached = llm.cached_prefix_len(prompt_ids)
                loop.call_soon_threadsafe(q.put_nowait, ("ready", cached))
                n, last = 0, None
                try:
                    for out in llm.stream(prompt_ids, sp, images):
                        if stop.is_set():
                            break
                        n, last = n + 1, out.token_id
                        if last != EOS_ID:
                            loop.call_soon_threadsafe(q.put_nowait, ("tok", out.token_id))
                except Exception as e:  # noqa: BLE001 - surfaced to the client as a finish
                    loop.call_soon_threadsafe(q.put_nowait, ("err", repr(e)))
                reason = InferenceFinishReason.Stop if last == EOS_ID or n < sp.max_tokens else InferenceFinishReason.Length
                loop.call_soon_threadsafe(q.put_nowait, ("end", reason))

        threading.Thread(target=run, daemon=True).start()
        try:
            while True:
                kind, val = await q.get()
                if kind == "ready":
                    yield InferenceChunk.ready(prompt_usage=PromptUsage(prompt_tokens=len(prompt_ids), prompt_cache_hit_tokens=val))
                elif kind == "tok":
                    yield InferenceChunk.tokens([val])
                elif kind == "err":
                    raise RuntimeError(val)
                else:
                    yield InferenceChunk.finish(finish_reason=val)
                    return
        finally:
            stop.set()

    async def body(protocol: str, conv, model: str, prompt_ids, sp, images=None) -> AsyncIterator[str]:
        req_t, resp_t = PROTOCOLS[protocol]
        rid = str(uuid4())
        proc = StreamProcessor(req_t.chunk_generator(conv, rid, model), conv.parsing_options, tok)
        resp = None if conv.stream else resp_t(rid, model, int(time()), 0, 0)
        try:
            it = aiter(infer(prompt_ids, sp, images))
            while True:
                try:
                    chunk = await anext(it)
                except StopAsyncIteration:
                    outs = proc.finish()
                else:
                    outs = proc.push(chunk)
                for o in outs:
                    if conv.stream:
                        ev = resp_t.sse_event_name(o)
                        yield (f"event: {ev}\n" if ev else "") + f"data: {o.to_json()}\n\n"
                    else:
                        resp.accumulate(o)
                if proc.finished:
                    break
            if conv.stream:
                done = resp_t.sse_done_data()
                if done is not None:
                    yield f"data: {done}\n\n"
            else:
                yield resp.to_json()
        finally:
            proc.close()

    def route(protocol: str):
        async def handler(request: Request) -> Response:
            req_t, _ = PROTOCOLS[protocol]
            try:
                req = req_t(await request.body())
            except ValueError as e:
                return JSONResponse({"error": {"message": str(e), "type": "invalid_request_error"}}, status_code=400)
            model = req.model or "deepseek-flash"
            conv = req.convert(conversion_options())
            prompt_ids, images = encode_conversation(conv.conversation, llm.root)
            sp = sampling_from(conv, len(prompt_ids))
            out = body(protocol, conv, model, prompt_ids, sp, images)
            if conv.stream:
                return StreamingResponse(out, media_type="text/event-stream")
            return Response("".join([p async for p in out]), media_type="application/json")

        return handler

    for path, protocol in (("/v1/chat/completions", "chat_completions"), ("/v1/responses", "responses"), ("/v1/messages", "messages")):
        app.add_api_route(path, route(protocol), methods=["POST"])

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "deepseek-flash", "object": "model", "owned_by": "local"}]}

    @app.get("/stats")
    async def stats():
        return {**llm.last_stats, "expert_cache_GB": round(llm.weights.experts.used / 2**30, 2)}

    return app
