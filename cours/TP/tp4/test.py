import os
from dotenv import load_dotenv
from openai import OpenAI
from dataset_utils import load_csv



if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("API_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    messages = [{"role": "user", "content": "hello"}]

    response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.6,
            logprobs=True,  # Important
            top_logprobs=1
            )

    content = response.choices[0].message.content
    logprobs = response.choices[0].logprobs.content
    print(content)