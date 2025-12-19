
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from main import app, ChatResponse

client = TestClient(app)

# 1. Critical File Existence Checks
def test_critical_files_exist():
    """
    Ensures that critical project files exist.
    """
    root_dir = Path(__file__).parent.parent
    critical_files = [
        "main.py",
        "Dockerfile",
        "requirements.txt",
        "tests/"
    ]

    for file_name in critical_files:
        file_path = root_dir / file_name
        assert file_path.exists(), f"Critical file missing: {file_name}"

# 2. API Contract Tests
def test_chat_response_contract():
    """
    Verifies that the ChatResponse model maintains its contract.
    This protects the frontend from breaking changes in response structure.
    """
    # Verify the keys in the pydantic model schema
    schema = ChatResponse.model_json_schema()
    properties = schema.get("properties", {})

    expected_keys = {"session_id", "answer", "documents"}
    assert expected_keys.issubset(properties.keys()), \
        f"ChatResponse model missing keys: {expected_keys - properties.keys()}"

def test_api_endpoints_schema():
    """
    Verifies that the API endpoints expose the expected schema via OpenAPI.
    This acts as a contract test for the API surface.
    """
    response = client.get("/openapi.json")
    assert response.status_code == 200
    openapi_schema = response.json()

    # Check /chat endpoint response schema
    chat_post = openapi_schema["paths"]["/chat"]["post"]
    assert "responses" in chat_post
    assert "200" in chat_post["responses"]

    # Verify we are referencing the ChatResponse model or an equivalent structure
    # Note: FastAPI might wrap this in a $ref
    response_schema = chat_post["responses"]["200"]["content"]["application/json"]["schema"]

    if "$ref" in response_schema:
        ref_name = response_schema["$ref"].split("/")[-1]
        model_schema = openapi_schema["components"]["schemas"][ref_name]
        properties = model_schema["properties"]

        expected_keys = ["session_id", "answer", "documents"]
        for key in expected_keys:
            assert key in properties, f"/chat response missing key: {key}"

def test_stats_endpoint_contract():
    """
    Verifies the /stats endpoint returns the expected structure.
    """
    # Since /stats is not typed with a Pydantic model in main.py (it returns a dict),
    # we test the runtime response structure.

    # Mock dependencies if needed, but here we just check the response keys
    # We might need to mock get_api_key if authentication is required for this test client
    # But TestClient might bypass it depending on how it's set up in main.py or if we override it.

    # We'll override the dependency to be safe
    # Note: app.dependency_overrides relies on the function object.
    # We need to import get_api_key from main to override it correctly.
    from main import get_api_key
    app.dependency_overrides[get_api_key] = lambda: True

    try:
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()

        expected_keys = ["total_sessions", "total_messages", "current_active_sessions"]
        for key in expected_keys:
            assert key in data, f"/stats response missing key: {key}"
    finally:
        app.dependency_overrides = {}
