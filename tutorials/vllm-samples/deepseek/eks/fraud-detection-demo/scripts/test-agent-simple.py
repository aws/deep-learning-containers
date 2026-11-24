#!/usr/bin/env python3
"""
Simple test to see how to extract response from Strands Agent
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, AsyncIterable
from strands import Agent
from strands.models import Model
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec


class DeepSeekVLLMModel(Model):
    """Custom Model class for DeepSeek R1 on vLLM/EKS"""
    
    def __init__(
        self,
        base_url: str,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        temperature: float = 0.3,
        max_tokens: int = 500
    ):
        self.config = {
            "base_url": base_url,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
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
        raise NotImplementedError("Not needed for this test")
    
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterable[StreamEvent]:
        """Stream responses from DeepSeek via vLLM"""
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
        
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"DeepSeek API error: {error_text}")
                
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                
                print(f"[DEBUG] Raw content from DeepSeek: {content[:200]}...")
                
                yield {"messageStart": {"role": "assistant"}}
                yield {"contentBlockDelta": {"delta": {"text": content}}}
                yield {"messageStop": {"stopReason": "end_turn"}}


# Test
print("=" * 70)
print("🧪 Testing Agent Response Extraction")
print("=" * 70)
print()

model = DeepSeekVLLMModel(
    base_url="http://k8s-default-vllmdeep-066fc65d38-1002306241.us-west-2.elb.amazonaws.com",
    temperature=0.3,
    max_tokens=500
)

agent = Agent(model=model, tools=[], system_prompt="You are a helpful assistant.")

print("Calling agent with simple question...")
query = "What is 2+2? Answer briefly."
response = agent(query)

print()
print("=" * 70)
print("RESPONSE ANALYSIS")
print("=" * 70)
print(f"Type: {type(response)}")
print(f"Dir: {[x for x in dir(response) if not x.startswith('_')]}")
print()

# Try different extraction methods
print("1. str(response):")
print(f"   '{str(response)}'")
print()

print("2. repr(response):")
print(f"   '{repr(response)}'")
print()

if hasattr(response, 'text'):
    print("3. response.text:")
    print(f"   '{response.text}'")
else:
    print("3. response.text: DOES NOT EXIST")

if hasattr(response, 'messages'):
    print("4. response.messages:")
    print(f"   {response.messages}")
else:
    print("4. response.messages: DOES NOT EXIST")

if hasattr(response, 'content'):
    print("5. response.content:")
    print(f"   '{response.content}'")
else:
    print("5. response.content: DOES NOT EXIST")

if hasattr(response, '__dict__'):
    print("6. response.__dict__:")
    print(f"   {response.__dict__}")
else:
    print("6. response.__dict__: DOES NOT EXIST")

print()
print("=" * 70)
print("✅ Analysis complete")
print("=" * 70)
