#!/usr/bin/env python3
"""
MCP Server: Geolocation Risk Checker
Detects impossible travel and location anomalies
"""

import math
from datetime import datetime
from fastmcp import FastMCP

mcp = FastMCP("Geolocation Risk Checker")

# City coordinates (lat, lon) for distance calculation
CITY_COORDS = {
    "New York, NY": (40.7128, -74.0060),
    "Los Angeles, CA": (34.0522, -118.2437),
    "Chicago, IL": (41.8781, -87.6298),
    "Houston, TX": (29.7604, -95.3698),
    "Moscow, Russia": (55.7558, 37.6173),
    "London, UK": (51.5074, -0.1278),
    "Tokyo, Japan": (35.6762, 139.6503),
    "Dubai, UAE": (25.2048, 55.2708),
    "Mumbai, India": (19.0760, 72.8777),
    "Sydney, Australia": (-33.8688, 151.2093)
}


def haversine_distance(coord1, coord2):
    """Calculate distance between two coordinates in miles"""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    R = 3959  # Earth's radius in miles
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon / 2) ** 2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance


@mcp.tool()
def check_geolocation_risk(
    customer_id: str,
    current_location: str,
    previous_transaction_location: str,
    time_difference_minutes: int
) -> dict:
    """
    Detect impossible travel and location anomalies.
    
    Args:
        customer_id: Customer identifier
        current_location: Current transaction location
        previous_transaction_location: Previous transaction location
        time_difference_minutes: Time between transactions in minutes
    
    Returns:
        Travel feasibility analysis with risk score and distance
    """
    
    # Get coordinates
    current_coords = CITY_COORDS.get(current_location)
    previous_coords = CITY_COORDS.get(previous_transaction_location)
    
    if not current_coords or not previous_coords:
        return {
            "customer_id": customer_id,
            "risk_score": 50,
            "travel_feasible": "UNKNOWN",
            "reason": "Location coordinates not found in database",
            "current_location": current_location,
            "previous_location": previous_transaction_location,
            "recommendation": "MANUAL REVIEW - Unknown location"
        }
    
    # Calculate distance
    distance_miles = haversine_distance(previous_coords, current_coords)
    
    # Calculate required speed (mph)
    time_hours = time_difference_minutes / 60
    if time_hours == 0:
        required_speed = float('inf')
    else:
        required_speed = distance_miles / time_hours
    
    # Determine feasibility and risk
    if distance_miles < 50:
        # Same city/nearby
        travel_feasible = "YES"
        risk_score = 0
        reason = "Locations are in close proximity"
        recommendation = "APPROVE - Normal travel pattern"
    
    elif required_speed < 500:
        # Possible by plane
        travel_feasible = "YES"
        risk_score = 20
        reason = f"Travel feasible by air (distance: {distance_miles:.0f} miles, time: {time_hours:.1f} hours)"
        recommendation = "APPROVE - Feasible travel"
    
    elif required_speed < 700:
        # Tight but possible
        travel_feasible = "UNLIKELY"
        risk_score = 60
        reason = f"Highly unlikely travel pattern (required speed: {required_speed:.0f} mph)"
        recommendation = "REVIEW - Suspicious travel pattern"
    
    else:
        # Impossible travel
        travel_feasible = "NO"
        risk_score = 95
        reason = f"IMPOSSIBLE TRAVEL detected (distance: {distance_miles:.0f} miles in {time_difference_minutes} minutes, required speed: {required_speed:.0f} mph)"
        recommendation = "BLOCK - Impossible travel indicates fraud"
    
    return {
        "customer_id": customer_id,
        "travel_feasible": travel_feasible,
        "risk_score": risk_score,
        "distance_miles": round(distance_miles, 1),
        "time_difference_minutes": time_difference_minutes,
        "required_speed_mph": round(required_speed, 1) if required_speed != float('inf') else "N/A",
        "current_location": current_location,
        "previous_location": previous_transaction_location,
        "reason": reason,
        "recommendation": recommendation,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    mcp.run(transport="sse")
