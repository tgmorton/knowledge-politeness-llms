import os

from openai import OpenAI

# API key should be set via OPENAI_API_KEY environment variable
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
