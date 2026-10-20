"""Structured outputs: regex, JSON schema and EBNF, via SamplingParams(so_args=...)."""

import json

from deepinfer_mac import LLM, SamplingParams

llm = LLM()
schema = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"}},
    "required": ["city", "population_millions"],
}
cases = [
    ("Give the capital of France and its population in JSON.", {"schema_type": "json", "schema": json.dumps(schema)}),
    ("What is 17 * 23? Answer with digits only.", {"schema_type": "regex", "schema": r"[0-9]+"}),
    ("Is the sky blue? Answer yes or no.", {"schema_type": "ebnf", "schema": 'root ::= "yes" | "no"'}),
]
for question, so_args in cases:
    ids, _ = llm.chat_prompt([{"role": "user", "content": question}])
    out = llm.generate(prompt_token_ids=[ids], sampling_params=SamplingParams(temperature=0, max_tokens=64, so_args=so_args))[0]
    print(so_args["schema_type"], "->", llm.tokenizer.decode([o.token_id for o in out], skip_special_tokens=True))
