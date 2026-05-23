import time
from typing import AsyncIterator

import openai
from loguru import logger

from app.providers.llm_base import LLMBase


class MIMOLLM(LLMBase):
    def __init__(self, api_key: str, endpoint: str, model: str, timeout: int = 30):
        self.model = model
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint,
            timeout=timeout,
        )

    async def chat_sync(
        self, messages: list[dict], temperature: float = 0.2
    ) -> str:
        start = time.time()
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        latency_ms = (time.time() - start) * 1000
        usage = response.usage
        logger.info(
            "LLM call completed: model={}, latency_ms={}, prompt_tokens={}, completion_tokens={}",
            self.model,
            round(latency_ms, 1),
            usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None,
        )
        return response.choices[0].message.content or ""

    async def chat_stream(
        self, messages: list[dict], temperature: float = 0.2
    ) -> AsyncIterator[str]:
        start = time.time()
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
        latency_ms = (time.time() - start) * 1000
        logger.info("LLM stream completed: model={}, latency_ms={}", self.model, round(latency_ms, 1))
