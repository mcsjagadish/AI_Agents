import os
from openai import AzureOpenAI

endpoint = "https://maheshagent-1-resource.cognitiveservices.azure.com/"
model_name = "gpt-4.1-mini"
deployment = "gpt-4.1-mini"

#Azure AI Foundary/maheshagent_1/Models + endpoints/gpt-4.1-mini/key
subscription_key = "get key from the above path"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_completion_tokens=800,
    temperature=1.0,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    model=deployment
)

print(response.choices[0].message.content)