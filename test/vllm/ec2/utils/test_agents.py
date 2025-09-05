# test_vllm_agent.py
from openai import OpenAI
from strands import Agent
from strands_tools import calculator, current_time

from pydantic import BaseModel, Field
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


class AnalysisResult(BaseModel):
    """Analysis result structure"""

    summary: str = Field(description="Main summary of the analysis")
    key_points: list[str] = Field(description="Key points extracted")
    confidence: float = Field(description="Confidence score (0-1)", ge=0, le=1)


class OpenAIClient:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def chat(self, model, messages, temperature, max_tokens):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def test_direct_completion():
    """Test direct API calls to VLLM"""
    client = OpenAIClient(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )

    chat_response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "What are the main benefits of using VLLM for inference?",
            },
        ],
        temperature=0.7,
        max_tokens=512,
    )

    print("\n API Response:")
    print(chat_response.choices[0].message.content)
    return client


def main():
    try:
        # Test direct API first
        model = test_direct_completion()

        # Create agent with the model
        agent = Agent(model=model, tools=[calculator, current_time])

        print("\nAgent initialized successfully!")

        # Test 1: Basic Agent Interaction
        print("\nTest 1: Basic Agent Interaction")
        response = agent("What are the main benefits of using VLLM for inference?")
        print(f"Agent Response: {response}")

        # Test 2: Tool Usage
        print("\nTest 2: Tool Usage")
        tool_response = agent("What's the square root of 144 and what's the current time?")
        print(f"Tool Response: {tool_response}")

        # Test 3: Structured Output
        print("\nTest 3: Structured Output")
        analysis_prompt = """
        Analyze this technical concept:
        VLLM is a high-performance library for LLM inference and serving,
        featuring state-of-the-art scheduling and optimization techniques.
        """

        result = agent.structured_output(AnalysisResult, analysis_prompt)

        print("Analysis Results:")
        print(f"Summary: {result.summary}")
        print(f"Key Points: {result.key_points}")
        print(f"Confidence: {result.confidence}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logging.error(f"Detailed error: {e}", exc_info=True)


if __name__ == "__main__":
    print("Starting VLLM Agent Test...")
    main()
