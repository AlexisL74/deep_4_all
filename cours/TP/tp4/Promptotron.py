from dataclasses import dataclass
import numpy as np
from openai import OpenAI


class Promptotron :
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "openai/gpt-oss-120b"
        self.low_temperature = 0.1
        self.high_temperature = 0.9


    def prompt(self, content: str, temperature: float) :
        messages = [{"role": "user", "content": content}]
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=temperature,
            logprobs=True,
            top_logprobs=1
            )
        
        output = response.choices[0].message.content
        logprobs_data = response.choices[0].logprobs

        tokens = []
        logprobs = []

        mean_logprob = np.exp(np.mean(logprobs))

        for token_info in logprobs_data.content:
            tokens.append(token_info.token)
            logprobs.append(token_info.logprob)

        return PrompteResponse(
            content,
            output,
            mean_logprob,
            temperature,
            tokens,
            len(tokens)
        )
        
    def low_temperature_prompt(self, content: str) :
        return self.prompt(content, self.low_temperature)

    def high_temperature_prompt(self, content: str) :
        return self.prompt(content, self.high_temperature)
    
@dataclass
class PrompteResponse :
    input: str
    output: str
    mean_logprob: float
    temperature: float
    tokens: list[str]
    num_tokens: int


__all__ = [ Promptotron, PrompteResponse ]