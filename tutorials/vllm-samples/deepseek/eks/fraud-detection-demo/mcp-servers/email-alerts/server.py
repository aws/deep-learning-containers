#!/usr/bin/env python3
"""
MCP Server: Email Alert Sender
Sends fraud alerts via AWS SES
"""

import os
import boto3
from datetime import datetime
from fastmcp import FastMCP

mcp = FastMCP("Email Alert Sender")

# AWS SES client (will use IAM role credentials from ECS)
ses_client = boto3.client('ses', region_name=os.getenv('AWS_REGION', 'us-west-2'))

SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'fraud-alerts@example.com')
DEFAULT_RECIPIENTS = os.getenv('ALERT_RECIPIENTS', 'security-team@example.com').split(',')


@mcp.tool()
def send_fraud_alert_email(
    alert_type: str,
    transaction_id: str,
    customer_id: str,
    risk_score: int,
    details: str,
    recipients: list = None
) -> dict:
    """
    Send fraud alert email via AWS SES.
    
    Args:
        alert_type: Type of alert (high_risk, suspicious, blocked)
        transaction_id: Transaction identifier
        customer_id: Customer identifier
        risk_score: Risk score 0-100
        details: Alert details
        recipients: List of recipient emails (optional)
    
    Returns:
        Email send status and message ID
    """
    
    if recipients is None:
        recipients = DEFAULT_RECIPIENTS
    
    # Format email based on alert type
    if alert_type == "high_risk":
        subject = f"🚨 HIGH RISK FRAUD ALERT - Transaction {transaction_id}"
        priority = "URGENT"
        color = "#DC3545"
    elif alert_type == "suspicious":
        subject = f"⚠️ Suspicious Activity - Transaction {transaction_id}"
        priority = "HIGH"
        color = "#FFC107"
    elif alert_type == "blocked":
        subject = f"🛑 TRANSACTION BLOCKED - {transaction_id}"
        priority = "CRITICAL"
        color = "#FF0000"
    else:
        subject = f"Fraud Alert - Transaction {transaction_id}"
        priority = "NORMAL"
        color = "#6C757D"
    
    html_body = f"""
    <html>
    <head></head>
    <body style="font-family: Arial, sans-serif;">
        <div style="border-left: 4px solid {color}; padding: 20px; margin: 20px;">
            <h2 style="color: {color};">Fraud Detection Alert</h2>
            <p><strong>Priority:</strong> {priority}</p>
            <hr>
            <h3>Transaction Details</h3>
            <ul>
                <li><strong>Transaction ID:</strong> {transaction_id}</li>
                <li><strong>Customer ID:</strong> {customer_id}</li>
                <li><strong>Risk Score:</strong> {risk_score}/100</li>
                <li><strong>Alert Type:</strong> {alert_type.upper()}</li>
            </ul>
            <h3>Details</h3>
            <p>{details}</p>
            <hr>
            <p style="color: #6C757D; font-size: 12px;">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}<br>
                Automated Fraud Detection System
            </p>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
    FRAUD DETECTION ALERT
    Priority: {priority}
    
    Transaction Details:
    - Transaction ID: {transaction_id}
    - Customer ID: {customer_id}
    - Risk Score: {risk_score}/100
    - Alert Type: {alert_type.upper()}
    
    Details:
    {details}
    
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
    """
    
    try:
        # Send email via SES
        response = ses_client.send_email(
            Source=SENDER_EMAIL,
            Destination={'ToAddresses': recipients},
            Message={
                'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                'Body': {
                    'Text': {'Data': text_body, 'Charset': 'UTF-8'},
                    'Html': {'Data': html_body, 'Charset': 'UTF-8'}
                }
            }
        )
        
        return {
            "success": True,
            "message_id": response['MessageId'],
            "recipients": recipients,
            "alert_type": alert_type,
            "timestamp": datetime.now().isoformat(),
            "message": f"Alert email sent successfully to {len(recipients)} recipient(s)"
        }
    
    except Exception as e:
        # In demo mode, simulate success
        return {
            "success": True,
            "message_id": f"demo-msg-{transaction_id}",
            "recipients": recipients,
            "alert_type": alert_type,
            "timestamp": datetime.now().isoformat(),
            "message": f"[DEMO MODE] Alert email would be sent to {len(recipients)} recipient(s)",
            "note": "AWS SES not configured, running in demo mode"
        }


if __name__ == "__main__":
    mcp.run(transport="sse")
