from openai import OpenAI

# Connect to the local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

# Send a simple chat message to the Qwen3 model
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B-AWQ",
    messages=[
        {"role": "user", "content": "Explain what vLLM is in one sentence."},
    ],
)

# Print the model's response
print(response.choices[0].message.content)