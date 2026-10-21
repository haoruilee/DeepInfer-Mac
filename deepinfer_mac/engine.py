"""DeepInfer-shaped offline engine: `LLM`, `SamplingParams`, per-token outputs.

Single-sequence, Mac-sized version of what DeepInfer does on a GPU cluster:
  * SWA Bounded Replay -- prefill runs the "encoder" (layers <= the last KV source) over the whole
    prompt but replays only the last `keep_tokens` (DeepInfer default 140) through the decoder
    layers, which see the prompt only through the shared compressed KV and a 128-token window.
  * chunked, expert-major prefill and prefix reuse across calls (multi-turn chat pays only for
    the new tokens).
  * structured output (regex / JSON schema / EBNF / JSON object) via xgrammar.
  * DSpark speculative decoding (engine/dspark.py), opt-in.

Environment knobs mirror DeepInfer's names where one exists:
  DEEPINFER_YOCO_PREFILL_KEEP_TOKENS (140), DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE (compact|off),
  DEEPINFER_ENABLE_MTP_VARLEN (0), DEEPINFER_USE_NEXTN_DECODE (5).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import numpy as np

from .model.engram import NgramHasher
from .model.model import Args, SeqState, Transformer
from .weights import DiskWeights

DEFAULT_MODEL_DIR = Path(os.environ.get("DEEPINFER_HF_MODEL_DIR", "~/models/dsv41-mac")).expanduser()
EOS_ID = 1


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 1024
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False
    seed: int | None = None
    seq_seed: int | None = None  # DeepInfer's name for the per-request seed
    so_args: dict | None = None  # {"schema_type": "json"|"regex"|"ebnf"|"json_object", "schema": str}


@dataclass
class RequestOutput:
    """One generated token, with the fields DeepInfer's examples read."""

    token_id: int
    token_logprobs: float
    prompt_len: int
    draft_tok: int = 0
    accept_tok: int = 0
    text: str = ""


# ------------------------------------------------------------------------------ sampling


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    return x - m - math.log(np.exp(x - m).sum())


def sample(logits: np.ndarray, sp: SamplingParams, rng: np.random.Generator, mask: np.ndarray | None = None) -> int:
    x = logits.astype(np.float64, copy=True)
    if mask is not None:
        x[~mask] = -np.inf
    if sp.temperature <= 0:
        return int(np.argmax(x))
    x /= sp.temperature
    if sp.top_k and sp.top_k < x.size:
        kth = np.partition(x, -sp.top_k)[-sp.top_k]
        x[x < kth] = -np.inf
    p = np.exp(x - x.max())
    p /= p.sum()
    if sp.top_p < 1.0:
        order = np.argsort(-p)
        c = np.cumsum(p[order])
        cut = order[np.searchsorted(c, sp.top_p) + 1 :]
        p[cut] = 0
        p /= p.sum()
    return int(rng.choice(p.size, p=p))


class Grammar:
    """xgrammar-backed constraint for one request."""

    _compiler = None

    def __init__(self, so_args: dict, tokenizer_json: Path, vocab_size: int):
        import xgrammar as xgr

        if Grammar._compiler is None:
            vocab = json.loads(Path(tokenizer_json).read_text())
            enc = [""] * vocab_size
            for tok, i in vocab["model"]["vocab"].items():
                if i < vocab_size:
                    enc[i] = tok
            for t in vocab.get("added_tokens", []):
                if t["id"] < vocab_size:
                    enc[t["id"]] = t["content"]
            info = xgr.TokenizerInfo(enc, vocab_type=xgr.VocabType.BYTE_LEVEL, vocab_size=vocab_size, stop_token_ids=[EOS_ID])
            Grammar._compiler = xgr.GrammarCompiler(info)
        c = Grammar._compiler
        kind, schema = so_args.get("schema_type", "json"), so_args.get("schema", "")
        if kind == "json":
            compiled = c.compile_json_schema(schema if isinstance(schema, str) else json.dumps(schema))
        elif kind == "json_object":
            compiled = c.compile_builtin_json_grammar()
        elif kind == "regex":
            compiled = c.compile_regex(schema)
        elif kind in ("ebnf", "grammar"):
            compiled = c.compile_grammar(schema)
        else:
            raise ValueError(f"unknown schema_type {kind!r}")
        self.xgr = xgr
        self.matcher = xgr.GrammarMatcher(compiled)
        self.vocab_size = vocab_size
        self.bitmask = xgr.allocate_token_bitmask(1, vocab_size)

    def mask(self) -> np.ndarray:
        self.matcher.fill_next_token_bitmask(self.bitmask)
        bits = self.bitmask.numpy().view(np.uint8)
        return np.unpackbits(bits, bitorder="little")[: self.vocab_size].astype(bool)

    def accept(self, tok: int) -> bool:
        return self.matcher.accept_token(tok)

    def done(self) -> bool:
        return self.matcher.is_terminated()


# ------------------------------------------------------------------------------ prompts

IMAGE_PLACEHOLDER = "<｜image｜>"  # what deepseek-recipe renders
IMAGE_TOKEN = "<｜deepseek_image｜>"  # how this checkpoint's tokenizer spells image_token_id
_recipe_tokenizers: dict = {}


def encode_conversation(conversation, root: Path) -> tuple[list[int], list]:
    """deepseek-recipe rendering -> token ids, plus the image sources in placeholder order. The
    image placeholder is spelled differently across tokenizer revisions (the reference takes its id
    from the config), so it is respelled here before encoding."""
    from deepseek_recipe import DeepseekV41Encoding, Tokenizer

    tok = _recipe_tokenizers.get(root) or _recipe_tokenizers.setdefault(root, Tokenizer.from_file(str(root / "tokenizer.json")))
    enc = DeepseekV41Encoding()
    rendered = enc.render_conversation(conversation)
    if not rendered.image_sources:
        return list(enc.with_tokenizer(tok).encode(conversation)), []
    ids = list(tok.encode(rendered.prompt.replace(IMAGE_PLACEHOLDER, IMAGE_TOKEN)))
    images = []
    for src in rendered.image_sources:
        if getattr(src, "data", None):
            images.append(bytes(src.data))
        else:
            images.append({"url": src.data_url or src.url})
    return ids, images


# ------------------------------------------------------------------------------ engine


class LLM:
    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL_DIR,
        expert_cache_gb: float | None = None,
        keep_tokens: int | None = None,
        bounded_replay: bool | None = None,
        prefill_chunk: int = 4096,
        speculative: bool | None = None,
        draft_len: int | None = None,
        moe_topk: int | None = None,
    ):
        self.root = root = Path(model).expanduser()
        mx.set_cache_limit(1 << 30)  # freed buffers go back to macOS instead of MLX's cache
        cfg = json.loads((root / "inference_config.json").read_text())
        self.args = args = Args.from_dict(cfg)
        if moe_topk:
            args.n_activated_experts = moe_topk
        budget = None if expert_cache_gb is None else int(expert_cache_gb * (1 << 30))
        self.weights = DiskWeights(root, args.dim, args.engram_layer_ids, budget)
        hasher = NgramHasher.load(args, root, cfg["engram_vocab_size"]) if args.engram_layer_ids else None
        self.model = Transformer(args, self.weights, hasher)
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        mode = os.environ.get("DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE", "compact")
        self.bounded_replay = (mode != "off") if bounded_replay is None else bounded_replay
        self.keep_tokens = keep_tokens or int(os.environ.get("DEEPINFER_YOCO_PREFILL_KEEP_TOKENS", "140"))
        assert self.keep_tokens >= args.window_size, "the retained suffix must cover the SWA window"
        self.prefill_chunk = max(prefill_chunk, self.keep_tokens)
        spec = os.environ.get("DEEPINFER_ENABLE_MTP_VARLEN", "0") == "1" if speculative is None else speculative
        self.draft_len = draft_len or int(os.environ.get("DEEPINFER_USE_NEXTN_DECODE", str(args.dspark_block_size)))
        self.dspark = None
        if spec:
            from .dspark import DSpark

            self.dspark = DSpark(args, self.weights, self.model)
        self._state: SeqState | None = None
        self.last_stats: dict = {}

    # -- prefill

    def _prefill(self, prompt: list[int], want_hidden: bool):
        """Feed `prompt`, reusing the cached sequence when it is a prefix. Returns (state,
        last logits, main hidden of the last position or None, number of reused tokens)."""
        st = self._state
        if st is not None and len(st.tokens) < len(prompt) and prompt[: len(st.tokens)] == st.tokens:
            reused = len(st.tokens)
        else:
            st, reused = self.model.new_state(), 0
        new = prompt[reused:]
        # chunking: the last chunk carries at least keep_tokens so bounded replay sees them all
        chunks, i = [], 0
        while len(new) - i > self.prefill_chunk:
            cut = min(self.prefill_chunk, len(new) - i - self.keep_tokens)
            chunks.append(new[i : i + cut])
            i += cut
        chunks.append(new[i:])
        logits = hidden = None
        for ci, ch in enumerate(chunks):
            last = ci == len(chunks) - 1
            if not self.bounded_replay:
                replay = None
            elif not last:
                replay = len(ch)  # decoder skipped entirely for this chunk
            else:
                replay = max(0, len(ch) - self.keep_tokens)
            out = self.model.forward(st, ch, replay_from=replay, want_hidden=want_hidden and last)
            if last:
                logits, hidden = out if want_hidden else (out, None)
        return st, logits, hidden, reused

    def _prefill_vl(self, prompt: list[int], images: list, want_hidden: bool):
        """A prompt with images: spans expanded, one chunk, no prefix reuse (image positions all
        carry the same token id, so token-level prefix matching cannot tell images apart)."""
        from .model.vision import expand_images

        toks, types, inputs = expand_images(prompt, images, self.args)
        st = self.model.new_state()
        replay = max(0, len(toks) - self.keep_tokens) if self.bounded_replay else None
        out = self.model.forward(st, toks, replay_from=replay, want_hidden=want_hidden, images=inputs, token_types=types)
        logits, hidden = out if want_hidden else (out, None)
        st.tokens[:] = [-1] * len(st.tokens)  # never reused as a text prefix
        return st, logits, hidden, 0

    # -- generation

    def stream(self, prompt_token_ids: list[int], sp: SamplingParams | None = None, images: list | None = None) -> Iterator[RequestOutput]:
        """`images`: one per image placeholder token in the prompt (bytes, path, URL, or an
        OpenAI/Anthropic image record)."""
        sp = sp or SamplingParams()
        seed = sp.seq_seed if sp.seq_seed is not None else sp.seed
        rng = np.random.default_rng(seed)
        grammar = Grammar(sp.so_args, self.root / "tokenizer.json", self.args.vocab_size) if sp.so_args else None
        stops = set(sp.stop_token_ids) | (set() if sp.ignore_eos else {EOS_ID})
        t0 = time.perf_counter()
        io0 = dict(self.weights.experts.stats)
        if images:
            st, logits, hidden, reused = self._prefill_vl(list(prompt_token_ids), images, want_hidden=self.dspark is not None)
        else:
            st, logits, hidden, reused = self._prefill(list(prompt_token_ids), want_hidden=self.dspark is not None)
        t_prefill = time.perf_counter() - t0
        n_prompt = len(prompt_token_ids)
        produced = 0
        if self.dspark is not None:
            gen = self.dspark.generate(self, st, logits, hidden, sp, rng, grammar, stops, n_prompt)
        else:
            gen = self._plain_decode(st, logits, sp, rng, grammar, stops, n_prompt)
        t1 = time.perf_counter()
        try:
            for out in gen:
                produced += 1
                yield out
        finally:
            self._state = st
            io = self.weights.experts.stats
            dt = time.perf_counter() - t1
            self.last_stats = {
                "prompt_tokens": n_prompt, "reused_tokens": reused, "prefill_s": round(t_prefill, 2),
                "prefill_tok_s": round((n_prompt - reused) / max(t_prefill, 1e-9), 1),
                "output_tokens": produced, "decode_s": round(dt, 2), "decode_tok_s": round(produced / max(dt, 1e-9), 2),
                "expert_hit_rate": round(self.weights.experts.hit_rate(), 3),
                "expert_GB_read": round((io["bytes"] - io0["bytes"]) / 1e9, 2),
            }

    def _plain_decode(self, st, logits, sp, rng, grammar, stops, n_prompt):
        for i in range(sp.max_tokens):
            lg = np.array(logits[-1], dtype=np.float32)
            tok = sample(lg, sp, rng, grammar.mask() if grammar else None)
            lp = float(log_softmax(lg)[tok])
            if grammar is not None:
                grammar.accept(tok)
            yield RequestOutput(tok, lp, n_prompt, text=self.tokenizer.decode([tok], skip_special_tokens=False))
            if tok in stops or (grammar is not None and grammar.done()) or i == sp.max_tokens - 1:
                return  # the last token is not fed: that forward would only serve a token never sampled
            logits = self.model.forward(st, [tok])

    def generate(
        self,
        prompts: list[str] | str | None = None,
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        prompt_token_ids: list[list[int]] | None = None,
        use_tqdm: bool = False,
        **_,
    ) -> list[list[RequestOutput]]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if prompt_token_ids is None:
            prompt_token_ids = [self.encode(p) for p in prompts]
        sps = sampling_params if isinstance(sampling_params, list) else [sampling_params or SamplingParams()] * len(prompt_token_ids)
        return [list(self.stream(ids, sp)) for ids, sp in zip(prompt_token_ids, sps)]

    def encode(self, text: str) -> list[int]:
        """Raw text -> ids with BOS (special tokens in the text are honoured)."""
        return [0] + self.tokenizer.encode(text, add_special_tokens=False).ids

    def chat_prompt(self, messages: list[dict], thinking: bool = False, **request) -> tuple[list[int], object]:
        """Render an OpenAI-style conversation exactly as DeepSeek's API does (deepseek-recipe)."""
        ids, images, conv = self.chat_inputs(messages, thinking, **request)
        if images:
            raise ValueError("the conversation has images: use chat_inputs() and stream(ids, sp, images)")
        return ids, conv

    def chat_inputs(self, messages: list[dict], thinking: bool = False, **request):
        """-> (prompt ids, images, converted request). Images are returned in placeholder order,
        ready for `stream(ids, sp, images)`."""
        from deepseek_recipe import ChatCompletionRequest, ConversionOptions, ModelVersion

        body = {"model": "deepseek-flash", "messages": messages, "thinking": {"type": "enabled" if thinking else "disabled"}, **request}
        conv = ChatCompletionRequest(json.dumps(body).encode()).convert(
            ConversionOptions().with_model_alias("deepseek-flash", ModelVersion.DeepseekV41Flash)
        )
        return (*encode_conversation(conv.conversation, self.root), conv)

    def cached_prefix_len(self, prompt: list[int]) -> int:
        st = self._state
        if st is not None and len(st.tokens) < len(prompt) and prompt[: len(st.tokens)] == st.tokens:
            return len(st.tokens)
        return 0

    def reset(self):
        self._state = None

    def close(self):
        self._state = None
