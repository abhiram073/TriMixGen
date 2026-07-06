import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend.config import settings
import asyncio

client = TestClient(app)
API_V1_STR = settings.API_V1_STR

def test_health_check():
    response = client.get(f"{API_V1_STR}/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "generator_loaded" in data

def test_model_info():
    response = client.get(f"{API_V1_STR}/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "TriMixGen-v1.0 (GEN_003)"

def test_generate_endpoint_success():
    payload = {
        "prompt": "Hello world",
        "style": "positive",
        "english_usage": "high",
        "temperature": 0.8
    }
    response = client.post(f"{API_V1_STR}/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert "language_tags" in data
    assert "cmi" in data

def test_generate_endpoint_empty_prompt():
    payload = {
        "prompt": "   ",
        "style": "positive"
    }
    response = client.post(f"{API_V1_STR}/generate", json=payload)
    assert response.status_code == 422 # Validation Error

def test_generate_endpoint_long_prompt():
    payload = {
        "prompt": "A" * 600,
        "style": "positive"
    }
    response = client.post(f"{API_V1_STR}/generate", json=payload)
    assert response.status_code == 422 # Validation Error

def test_tag_endpoint_success():
    payload = {
        "text": "Idi chala bagundi andi"
    }
    response = client.post(f"{API_V1_STR}/tag", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert "labels" in data
    assert "cmi" in data

def test_tag_endpoint_empty():
    payload = {
        "text": ""
    }
    response = client.post(f"{API_V1_STR}/tag", json=payload)
    assert response.status_code == 422 # Validation Error
