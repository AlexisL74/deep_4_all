import asyncio
from dataclasses import dataclass
import numpy as np
from openai import OpenAI, AsyncOpenAI


class Promptotron :
    def __init__(self, api_key: str, base_url: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = "openai/gpt-oss-120b"
        self.low_temperature = 0.1
        self.high_temperature = 0.9


    async def prompt(self, content: str, temperature: float, id: int) :
        messages = [{"role": "user", "content": content}]
        response = await self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=temperature,
            logprobs=True,
            top_logprobs=1
            )
        
        output = response.choices[0].message.content
        logprobs_data = response.choices[0].logprobs

        logprobs = []

        for token_info in logprobs_data.content:
            logprobs.append(token_info.logprob)

        mean_logprob = np.exp(np.mean(logprobs))

        return PrompteResponse(
            id,
            content,
            output,
            mean_logprob,
            temperature,
        )
        
    async def low_temperature_prompt(self, content: str, id: int) :
        return await self.prompt(content, self.low_temperature, id)

    async def high_temperature_prompt(self, content: str, id: int) :
        return await self.prompt(content, self.high_temperature, id)
    
    async def double_temperature_prompt_async(self, content: str, id: int):
        return await asyncio.gather(
            self.low_temperature_prompt(content, id),
            self.high_temperature_prompt(content, id)
        )

    def double_temperature_prompt(self, content: str, id: int):
        return asyncio.run(
            self.double_temperature_prompt_async(content, id)
        )
    
@dataclass
class PrompteResponse :
    id: int
    input: str
    output: str
    mean_logprob: float
    temperature: float


__all__ = [ Promptotron, PrompteResponse ]