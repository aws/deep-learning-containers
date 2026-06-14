#!/usr/bin/env python3
"""
MCP Server: Transaction Risk Checker
Analyzes financial transactions for fraud indicators
"""

import json
import random
from datetime import datetime
from fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("Transaction Risk Checker")

# Mock database of high-risk merchants and locations
HIGH_RISK_MERCHANTS = {
    "ONLINE-STORE-RU.com": {"risk_factor": 0.8, "category": "cross_border_online"},
    "CRYPTO-EXCHANGE-XX": {"risk_factor": 0.9, "category": "cryptocurrency"},
    "GAMBLING-SITE-YY": {"risk_factor": 0.75, "category": "gambling"},
    "WIRE-TRANSFER-ZZ": {"risk_factor": 0.85, "category": "money_transfer"},
}

HIGH_RISK_LOCATIONS = ["Russia", "Nigeria", "North Korea", "Iran", "Venezuela"]


@mcp.tool()
def check_transaction_risk(
    transaction_id: str,
    customer_id: str,
    amount: float,
    merchant: str,
    location: str,
    transaction_time: str,
    card_present: bool = False
) -> dict:
    """
    Analyze a transaction for fraud indicators and calculate risk score.
    
    Args:
        transaction_id: Unique transaction identifier
        customer_id: Customer identifier
        amount: Transaction amount in USD
        merchant: Merchant name
        location: Transaction location
        transaction_time: Transaction timestamp
        card_present: Whether card was physically present
    
    Returns:
        Risk analysis with score (0-100), indicators, and recommendation
    """
    
    risk_score = 0
    risk_indicators = []
    
    # Check amount thresholds
    if amount > 5000:
        risk_score += 30
        risk_indicators.append(f"High transaction amount: ${amount:,.2f}")
    elif amount > 2000:
        risk_score += 15
        risk_indicators.append(f"Elevated transaction amount: ${amount:,.2f}")
    
    # Check merchant risk
    merchant_info = HIGH_RISK_MERCHANTS.get(merchant)
    if merchant_info:
        merchant_risk = int(merchant_info["risk_factor"] * 40)
        risk_score += merchant_risk
        risk_indicators.append(f"High-risk merchant: {merchant} (category: {merchant_info['category']})")
    
    # Check location risk
    if any(risky_loc in location for risky_loc in HIGH_RISK_LOCATIONS):
        risk_score += 25
        risk_indicators.append(f"High-risk location: {location}")
    
    # Check card presence
    if not card_present and amount > 1000:
        risk_score += 15
        risk_indicators.append("Card not present for high-value transaction")
    
    # Check time (transactions late at night/early morning are riskier)
    try:
        hour = int(transaction_time.split(":")[0].split()[-1])
        if hour >= 0 and hour < 6:
            risk_score += 10
            risk_indicators.append(f"Unusual transaction time: {transaction_time}")
    except:
        pass
    
    # Cap at 100
    risk_score = min(risk_score, 100)
    
    # Determine risk category and recommendation
    if risk_score >= 80:
        risk_category = "CRITICAL"
        recommendation = "BLOCK TRANSACTION - High fraud probability"
    elif risk_score >= 60:
        risk_category = "HIGH"
        recommendation = "MANUAL REVIEW REQUIRED - Multiple risk factors"
    elif risk_score >= 40:
        risk_category = "MEDIUM"
        recommendation = "ENHANCED VERIFICATION - Monitor closely"
    elif risk_score >= 20:
        risk_category = "LOW"
        recommendation = "APPROVE WITH MONITORING"
    else:
        risk_category = "MINIMAL"
        recommendation = "APPROVE TRANSACTION"
    
    return {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "risk_indicators": risk_indicators,
        "recommendation": recommendation,
        "analysis_timestamp": datetime.now().isoformat(),
        "amount": amount,
        "merchant": merchant,
        "location": location
    }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="sse")
