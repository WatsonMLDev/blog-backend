import os
import pytest
from unittest.mock import MagicMock, patch

# Set dummy env vars before importing pipeline to avoid KeyErrors
os.environ["GEMINI_API_KEY"] = "fake_key"
os.environ["DOCUMENT_STORE_PATH"] = "./fake_path"

from src.pipeline import PortfolioRagPipeline

@pytest.fixture
def mock_pipeline():
    with patch('src.pipeline.ChromaDocumentStore'), \
         patch('src.pipeline.GoogleGenAITextEmbedder'), \
         patch('src.pipeline.ChromaEmbeddingRetriever'), \
         patch('src.pipeline.GoogleGenAIChatGenerator'):
        return PortfolioRagPipeline()

def test_intent_prompt_structure(mock_pipeline):
    """Verify Intent prompt uses System and User roles."""
    res = mock_pipeline.intent_prompt_builder.run(question="Hi", chat_history="User: Hello")
    messages = res['prompt']

    assert len(messages) == 2
    assert messages[0].role.value == "system"
    assert "Classify user intent" in messages[0].text
    assert messages[1].role.value == "user"
    assert "User message: Hi" in messages[1].text

def test_chat_prompt_structure(mock_pipeline):
    """Verify Chat prompt uses System and User roles."""
    res = mock_pipeline.chat_prompt_builder.run(question="How are you?")
    messages = res['prompt']

    assert len(messages) == 2
    assert messages[0].role.value == "system"
    assert "specific purpose portfolio assistant" in messages[0].text
    assert messages[1].role.value == "user"
    assert "How are you?" in messages[1].text

def test_rag_prompt_structure(mock_pipeline):
    """Verify RAG prompt contains XML tags and context."""
    mock_docs = [MagicMock(content="Test Content", meta={'file_name': 'test.md'})]
    res = mock_pipeline.rag_prompt_builder.run(
        question="Q?",
        expanded_query="Q?",
        documents=mock_docs
    )
    messages = res['prompt']

    assert len(messages) == 2
    assert messages[0].role.value == "system"
    assert messages[1].role.value == "user"

    content = messages[1].text
    assert "<question>Q?</question>" in content
    assert "<context>" in content
    assert "Test Content" in content
