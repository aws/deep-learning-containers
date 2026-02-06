#!/usr/bin/env python3
"""
MCP Server: Customer Identity Verifier
Multi-factor identity verification for fraud prevention
"""

import hashlib
from datetime import datetime
from fastmcp import FastMCP

mcp = FastMCP("Customer Identity Verifier")

# Mock customer database
KNOWN_CUSTOMERS = {
    "C-12345": {
        "name": "John Smith",
        "device_fingerprints": ["dev-abc123", "dev-xyz789"],
        "known_ips": ["192.168.1.100", "10.0.0.50"],
        "biometric_hash": hashlib.sha256("biometric_data_1".encode()).hexdigest()
    },
    "C-67890": {
        "name": "Sarah Johnson", 
        "device_fingerprints": ["dev-def456"],
        "known_ips": ["172.16.0.20"],
        "biometric_hash": hashlib.sha256("biometric_data_2".encode()).hexdigest()
    },
    "C-45678": {
        "name": "Mike Chen",
        "device_fingerprints": ["dev-ghi789"],
        "known_ips": ["192.168.2.15"],
        "biometric_hash": hashlib.sha256("biometric_data_3".encode()).hexdigest()
    }
}


@mcp.tool()
def verify_customer_identity(
    customer_id: str,
    device_fingerprint: str,
    ip_address: str,
    biometric_data: str = None
) -> dict:
    """
    Verify customer identity using multiple factors.
    
    Args:
        customer_id: Customer identifier
        device_fingerprint: Device fingerprint hash
        ip_address: IP address of the request
        biometric_data: Optional biometric verification data
    
    Returns:
        Verification result with confidence score and matched factors
    """
    
    if customer_id not in KNOWN_CUSTOMERS:
        return {
            "customer_id": customer_id,
            "verification_status": "UNKNOWN_CUSTOMER",
            "confidence_score": 0,
            "matched_factors": [],
            "failed_factors": ["customer_id"],
            "recommendation": "DENY - Unknown customer",
            "timestamp": datetime.now().isoformat()
        }
    
    customer = KNOWN_CUSTOMERS[customer_id]
    matched_factors = []
    failed_factors = []
    confidence_score = 0
    
    # Check device fingerprint (30 points)
    if device_fingerprint in customer["device_fingerprints"]:
        matched_factors.append("device_fingerprint")
        confidence_score += 30
    else:
        failed_factors.append("device_fingerprint")
    
    # Check IP address (25 points)
    if ip_address in customer["known_ips"]:
        matched_factors.append("ip_address")
        confidence_score += 25
    else:
        failed_factors.append("ip_address")
    
    # Check biometric data (45 points)
    if biometric_data:
        bio_hash = hashlib.sha256(biometric_data.encode()).hexdigest()
        if bio_hash == customer["biometric_hash"]:
            matched_factors.append("biometric")
            confidence_score += 45
        else:
            failed_factors.append("biometric")
    
    # Determine status
    if confidence_score >= 70:
        status = "VERIFIED"
        recommendation = "APPROVE - High confidence match"
    elif confidence_score >= 50:
        status = "PARTIAL_MATCH"
        recommendation = "REVIEW - Additional verification recommended"
    else:
        status = "FAILED"
        recommendation = "DENY - Insufficient verification factors"
    
    return {
        "customer_id": customer_id,
        "customer_name": customer["name"],
        "verification_status": status,
        "confidence_score": confidence_score,
        "matched_factors": matched_factors,
        "failed_factors": failed_factors,
        "recommendation": recommendation,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    mcp.run(transport="sse")
