from autogen import UserProxyAgent, ConversableAgent

local_llm_config = {
    "config_list": [
        {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # Same as in vLLM command
            "api_key": "NotRequired",  # Not needed
            "base_url": "http://localhost:8000/v1",  # Your vLLM URL, with '/v1' added
        }
    ],
    "cache_seed": None,  # Turns off caching, useful for testing different models
}

# Create the agent that uses the LLM.
assistant = ConversableAgent("agent", llm_config=local_llm_config, system_message="")

# Create the agent that represents the user in the conversation.
user_proxy = UserProxyAgent("user", code_execution_config=False, system_message="")

# Let the assistant start the conversation.  It will end when the user types exit.
assistant.initiate_chat(user_proxy, message="How can I help you today?")
