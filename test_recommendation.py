import os
import pytest
from fastapi.testclient import TestClient
from src.inference.api import app

client = TestClient(app)

# Test recommendation endpoint
def test_recommendation_endpoint():
    response = client.post(
        "/recommend",
        json={
            "customer_id": "12345",
            "recent_service_ids": ["service1", "service2"],
            "top_k": 5
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 5

# Test health endpoint
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

# Test missing customer ID
def test_missing_customer_id():
    response = client.post(
        "/recommend",
        json={"recent_service_ids": ["service1"]}
    )
    assert response.status_code == 422

# Test invalid top_k
def test_invalid_top_k():
    response = client.post(
        "/recommend",
        json={
            "customer_id": "12345",
            "recent_service_ids": ["service1"],
            "top_k": "invalid"
        }
    )
    assert response.status_code == 422