
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from main import app, get_api_key

client = TestClient(app)

@pytest.fixture(autouse=True)
def override_api_key():
    app.dependency_overrides[get_api_key] = lambda: True
    yield
    app.dependency_overrides = {}

@pytest.fixture
def mock_rag_pipeline():
    with patch("main.rag_pipeline") as mock:
        yield mock

@pytest.fixture
def mock_stats_tracker():
    with patch("main.stats_tracker") as mock:
        yield mock

@pytest.fixture
def mock_session_manager():
    with patch("main.session_manager") as mock:
        yield mock

def test_inference_stats_logging(mock_rag_pipeline, mock_stats_tracker, mock_session_manager):
    """
    Test that stats_tracker.log_event is called with the correct metadata
    after a successful chat request.
    """
    # Mock pipeline response with metadata
    mock_rag_pipeline.run = AsyncMock(return_value={
        "answer": "Test answer",
        "documents": [],
        "intent": "search",
        "latency": 1.23,
        "model": "gemini-2.5-flash"
    })

    # Mock session manager to return a valid session so a new one isn't created
    mock_session = AsyncMock()
    mock_session_manager.get_session.return_value = mock_session
    mock_session_manager.create_session.return_value = "test-session"

    # Make request
    response = client.post(
        "/chat",
        json={"question": "test question", "session_id": "test-session"}
    )

    assert response.status_code == 200

    # Verify log_event was called for inference_completed
    # We expect multiple calls (session_created, message_sent, inference_completed)
    # Let's find the inference_completed call

    calls = mock_stats_tracker.log_event.call_args_list
    inference_call = None
    for call in calls:
        if call[0][0] == "inference_completed":
            inference_call = call
            break

    assert inference_call is not None

    # Check arguments
    args, _ = inference_call
    event_type, session_id, metadata = args

    assert event_type == "inference_completed"
    assert session_id == "test-session"
    assert metadata["latency"] == 1.23
    assert metadata["intent"] == "search"
    assert metadata["model"] == "gemini-2.5-flash"
    assert metadata["doc_count"] == 0
