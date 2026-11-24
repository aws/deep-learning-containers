#!/usr/bin/env python3
"""Test to extract message content properly"""

# Based on test results, response.message has the structure
# Let's see if we can extract text from it

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, AsyncIterable
from strands import Agent
from strands.models import Model
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec


class DeepSeekVLLMModel(Model):
    def __init__(self, base_url: str, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", temperature: float = 0.3, max_tokens: int = 500):
        self.config = {"base_url": base_url, "model_name": model_name, "temperature": temperature, "max_tokens": max_tokens}
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_endpoint = f"{base_url}/v1/chat/completions"
    
    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()
    
    def update_config(self, **kwargs) -> None:
        self.config.update(kwargs)
    
    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError()
    
    async def stream(self, messages: Messages, tool_specs: Optional[List[ToolSpec]] = None, system_prompt: Optional[str] = None, **kwargs: Any) -> AsyncIterable[StreamEvent]:
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            role = msg.get("role", "user")
            content_blocks = msg.get("content", [])
            text_content = ""
            for block in content_blocks:
                if "text" in block:
                    text_content += block["text"]
            if text_content:
                formatted_messages.append({"role": role, "content": text_content})
        
        payload = {"model": self.model_name, "messages": formatted_messages, "temperature": self.temperature, "max_tokens": self.max_tokens, "stream": False}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"DeepSeek API error: {error_text}")
                
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                
                yield {"messageStart": {"role": "assistant"}}
                yield {"contentBlockDelta": {"delta": {"text": content}}}
                yield {"messageStop": {"stopReason": "end_turn"}}


model = DeepSeekVLLMModel(base_url="http://k8s-default-vllmdeep-066fc65d38-1002306241.us-west-2.elb.amazonaws.com", temperature=0.3, max_tokens=200)
agent = Agent(model=model, tools=[], system_prompt="You are helpful.")

print("Calling agent...")
response = agent("What is 2+2? Be brief.")

print(f"\nresponse.message: {response.message}")
print(f"\nresponse.message['content']: {response.message.get('content', 'NO CONTENT KEY')}")

# Try to extract text
if response.message and 'content' in response.message:
    content_blocks = response.message['content']
    text = ""
    for block in content_blocks:
        if isinstance(block, dict) and 'text' in block:
            text += block['text']
        elif isinstance(block, str):
            text += block
    print(f"\nExtracted text: '{text}'")
else:
    print("\nCould not extract - content is empty or missing!")
