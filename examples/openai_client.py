"""Talk to `python -m deepinfer_mac serve` with the official OpenAI SDK (pip install openai)."""

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
stream = client.chat.completions.create(
    model="deepseek-flash",
    messages=[{"role": "user", "content": "Explain mixture-of-experts in two sentences."}],
    stream=True,
    max_tokens=200,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
