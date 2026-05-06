#!/usr/bin/env python3
"""
Financial Fraud Detection UI
Real-time fraud analysis with AI agent and MCP tools
"""

import streamlit as st
import os
import sys
import io
import json
from datetime import datetime
from strands import Agent
from strands.models import Model
from mcp.client.sse import sse_client
from strands.tools.mcp import MCPClient
from typing import List, Dict, Any, Optional, AsyncIterable
import aiohttp
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-critical {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-high {
        background-color: #ff8800;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffcc00;
        color: black;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-low {
        background-color: #88cc00;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .tool-call {
        background-color: #f0f0f0;
        padding: 10px;
        border-left: 3px solid #1f77b4;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# DeepSeek vLLM Model class (from working old/agent.py)
class DeepSeekVLLMModel(Model):
    """Custom Model class for DeepSeek R1 on vLLM/EKS"""
    
    def __init__(
        self,
        base_url: str,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        temperature: float = 0.3,
        max_tokens: int = 1000
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
        if "base_url" in kwargs:
            self.base_url = kwargs["base_url"]
            self.api_endpoint = f"{self.base_url}/v1/chat/completions"
        if "model_name" in kwargs:
            self.model_name = kwargs["model_name"]
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        if "max_tokens" in kwargs:
            self.max_tokens = kwargs["max_tokens"]
    
    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError("Structured output not implemented for DeepSeek vLLM model")
    
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterable[StreamEvent]:
        """Stream responses from DeepSeek via vLLM"""
        # Convert Strands messages to OpenAI format
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
                
                yield {"messageStart": {"role": "assistant"}}
                yield {"contentBlockDelta": {"delta": {"text": content}}}
                yield {"messageStop": {"stopReason": "end_turn"}}


# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []


@st.cache_resource
def initialize_agent():
    """Initialize the fraud detection agent with MCP tools"""
    
    # MCP server endpoints from environment variables
    mcp_endpoints = {
        "transaction_risk": os.getenv("MCP_TRANSACTION_RISK_URL", "http://transaction-risk.fraud-detection.local:8080/sse"),
        "identity_verifier": os.getenv("MCP_IDENTITY_VERIFIER_URL", "http://identity-verifier.fraud-detection.local:8080/sse"),
        "email_alerts": os.getenv("MCP_EMAIL_ALERTS_URL", "http://email-alerts.fraud-detection.local:8080/sse"),
        "fraud_logger": os.getenv("MCP_FRAUD_LOGGER_URL", "http://fraud-logger.fraud-detection.local:8080/sse"),
        "geolocation": os.getenv("MCP_GEOLOCATION_URL", "http://geolocation-checker.fraud-detection.local:8080/sse"),
        "report_generator": os.getenv("MCP_REPORT_GENERATOR_URL", "http://report-generator.fraud-detection.local:8080/sse")
    }
    
    # Connect to all MCP servers
    mcp_clients = {}
    all_tools = []
    
    for name, endpoint in mcp_endpoints.items():
        try:
            client = MCPClient(lambda e=endpoint: sse_client(e))
            mcp_clients[name] = client
            # Get tools from this server
            with client:
                tools = client.list_tools_sync()
                all_tools.extend(tools)
        except Exception as e:
            st.warning(f"Could not connect to {name} MCP server: {e}")
    
    # Create DeepSeek model on EKS
    vllm_endpoint = os.getenv("VLLM_ENDPOINT")
    if not vllm_endpoint:
        raise ValueError("VLLM_ENDPOINT environment variable must be set to your vLLM ALB endpoint")
    
    model = DeepSeekVLLMModel(
        base_url=vllm_endpoint,
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        temperature=0.3,
        max_tokens=3500
    )
    
    # System prompt
    system_prompt = """You are an AI fraud detection specialist for a financial institution.

You have access to specialized tools for fraud detection. When analyzing transactions:
1. Check transaction risk factors
2. Verify customer identity
3. Analyze geolocation patterns
4. Send alerts for high-risk cases
5. Log all investigations
6. Generate reports

Be thorough and use multiple tools to make informed decisions. 
If risk score > 70, always send alerts and log the case."""
    
    # Create agent
    agent = Agent(model=model, tools=all_tools, system_prompt=system_prompt)
    
    return agent, mcp_clients


# Main UI
st.markdown('<div class="main-header">🛡️ Financial Fraud Detection System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Display model info
    st.info("🤖 **AI Model**: DeepSeek R1 32B on EKS")
    
    st.divider()
    
    # Statistics
    st.header("📊 Today's Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Analyzed", len(st.session_state.analysis_history))
    with col2:
        fraud_count = sum(1 for h in st.session_state.analysis_history if h.get('risk_score', 0) > 70)
        st.metric("Blocked", fraud_count)
    
    st.divider()
    
    # Recent alerts
    st.header("🚨 Recent Alerts")
    if st.session_state.analysis_history:
        for item in st.session_state.analysis_history[-3:]:
            if item.get('risk_score', 0) > 70:
                st.error(f"🚨 {item.get('transaction_id', 'Unknown')}")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["🔍 Analyze Transaction", "📜 History", "📈 Reports"])

with tab1:
    st.header("Transaction Analysis")
    
    # Transaction input form
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_id = st.text_input("Transaction ID", value=f"TXN-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            customer_id = st.selectbox("Customer ID", ["C-12345", "C-67890", "C-45678", "C-99999"])
            amount = st.number_input("Amount ($)", min_value=0.0, value=4500.0, step=100.0)
            merchant = st.selectbox("Merchant", [
                "Apple Store - Fifth Avenue",
                "ONLINE-STORE-RU.com",
                "CRYPTO-EXCHANGE-XX",
                "Local Coffee Shop"
            ])
        
        with col2:
            location = st.selectbox("Location", [
                "New York, NY",
                "Los Angeles, CA",
                "Moscow, Russia",
                "London, UK",
                "Tokyo, Japan"
            ])
            transaction_time = st.text_input("Time", value=datetime.now().strftime("%H:%M:%S"))
            card_present = st.checkbox("Card Present")
            device_fingerprint = st.text_input("Device Fingerprint", value="dev-abc123")
        
        # Additional fields
        ip_address = st.text_input("IP Address", value="192.168.1.100")
        previous_location = st.selectbox("Previous Location (30 min ago)", [
            "New York, NY",
            "Los Angeles, CA",
            "Moscow, Russia"
        ])
        
        submit = st.form_submit_button("🔍 Analyze Transaction", type="primary")
    
    if submit:
        # Initialize agent if needed
        if st.session_state.agent is None:
            with st.spinner("Initializing AI agent and connecting to fraud detection tools..."):
                st.session_state.agent, mcp_clients = initialize_agent()
        
        # Create analysis query
        query = f"""Analyze this financial transaction for fraud:

Transaction Details:
- ID: {transaction_id}
- Customer: {customer_id}
- Amount: ${amount:,.2f}
- Merchant: {merchant}
- Location: {location}
- Time: {transaction_time}
- Card Present: {card_present}
- Device: {device_fingerprint}
- IP: {ip_address}
- Previous Location: {previous_location} (30 minutes ago)

Please analyze this transaction comprehensively using all available tools and provide a fraud risk assessment."""
        
        # Run analysis
        st.divider()
        st.subheader("🤖 AI Agent Analysis")
        
        with st.spinner("Analyzing transaction..."):
            try:
                # Capture stdout while agent runs
                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()
                
                try:
                    response = st.session_state.agent(query)
                finally:
                    sys.stdout = old_stdout
                
                # Get the captured text
                response_text = captured_output.getvalue()
                
                # Display response
                if response_text and response_text.strip():
                    with st.expander("📋 Full Analysis Report", expanded=True):
                        st.markdown(response_text)
                else:
                    st.warning("⚠️ Agent returned empty response")
                
                # Store in history
                st.session_state.analysis_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "transaction_id": transaction_id,
                    "customer_id": customer_id,
                    "amount": amount,
                    "risk_score": 85,  # Extract from response
                    "decision": "BLOCKED"
                })
                
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)

with tab2:
    st.header("Analysis History")
    
    if st.session_state.analysis_history:
        for item in reversed(st.session_state.analysis_history):
            with st.expander(f"{item['transaction_id']} - {item['timestamp']}", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Amount", f"${item['amount']:,.2f}")
                col2.metric("Risk Score", f"{item.get('risk_score', 0)}/100")
                col3.metric("Decision", item.get('decision', 'N/A'))
    else:
        st.info("No analysis history yet. Analyze a transaction to get started.")

with tab3:
    st.header("Fraud Detection Reports")
    st.info("📊 Report generation coming soon")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>🛡️ Financial Fraud Detection System | Powered by AI & MCP Tools</p>
    <p>DeepSeek R1 32B • vLLM on EKS • AWS Deep Learning Containers</p>
</div>
""", unsafe_allow_html=True)
