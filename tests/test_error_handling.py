
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from main import app, get_api_key
import google.genai.errors

client = TestClient(app)

# Mocking the API key validation using dependency_overrides
@pytest.fixture(autouse=True)
def override_api_key():
    app.dependency_overrides[get_api_key] = lambda: True
    yield
    app.dependency_overrides = {}

@pytest.fixture
def mock_rag_pipeline():
    with patch("main.rag_pipeline") as mock:
        yield mock

def test_google_genai_429_error(mock_rag_pipeline, caplog):
    """
    Test that a Google GenAI 429 error (wrapped in RuntimeError)
    results in a 429 Too Many Requests response with a simplified message.
    """
    # Create the Google GenAI ClientError
    genai_error = google.genai.errors.ClientError(
        code=429,
        response_json={
            "error": {
                "code": 429,
                "message": "You exceeded your current quota...",
                "status": "RESOURCE_EXHAUSTED"
            }
        },
        response=None
    )

    # Wrap it in a RuntimeError
    runtime_error = RuntimeError(f"Error in Google Gen AI chat generation: {genai_error}")
    runtime_error.__cause__ = genai_error

    mock_rag_pipeline.run = AsyncMock(side_effect=runtime_error)

    response = client.post(
        "/chat",
        json={"question": "test question"}
    )

    # Verify status code
    assert response.status_code == 429

    # Verify simplified error message
    assert response.json() == {"error": "Rate limit exceeded. Please try again later."}

    # Verify concise logging
    # We expect: "Google GenAI API Error: Code=429, Status=RESOURCE_EXHAUSTED, Message=Rate limit exceeded. Please try again later., Endpoint=/chat"
    assert "Google GenAI API Error: Code=429" in caplog.text
    assert "Rate limit exceeded" in caplog.text
    # Ensure the full verbose JSON is NOT in the logs (checking for a snippet of it)
    assert "You exceeded your current quota" not in caplog.text

def test_google_genai_400_error(mock_rag_pipeline):
    """Test handling of 400 Bad Request error."""
    genai_error = google.genai.errors.ClientError(
        code=400,
        response_json={"error": {"code": 400, "status": "INVALID_ARGUMENT"}},
        response=None
    )
    runtime_error = RuntimeError(f"Error: {genai_error}")
    runtime_error.__cause__ = genai_error

    mock_rag_pipeline.run = AsyncMock(side_effect=runtime_error)

    response = client.post("/chat", json={"question": "test"})

    assert response.status_code == 400
    assert response.json() == {"error": "Bad request. Please check your input."}

def test_generic_runtime_error(mock_rag_pipeline, caplog):
    """Test that unrelated RuntimeErrors still result in 500."""
    runtime_error = RuntimeError("Some other random error")
    mock_rag_pipeline.run = AsyncMock(side_effect=runtime_error)

    response = client.post("/chat", json={"question": "test"})

    assert response.status_code == 500
    assert response.json() == {"error": "Failed to generate answer."}
    assert "Chat endpoint error: Some other random error" in caplog.text
