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

def test_unified_prompt_structure(mock_pipeline):
    """Verify Unified Prompt (Fast Mode) structure."""
    mock_docs = [MagicMock(content="Test Content", meta={'file_name': 'test.md'})]
    res = mock_pipeline.unified_prompt_builder.run(question="Q?", documents=mock_docs)
    messages = res['prompt']

    assert len(messages) == 2
    assert messages[0].role.value == "system"
    assert "Unified Prompt" not in messages[0].text # Just checking content isn't leaked
    assert "INSTRUCTIONS:" in messages[0].text

    content = messages[1].text
    assert "<question>Q?</question>" in content
    assert "<context>" in content
    assert "Test Content" in content

def test_fast_mode_execution():
    """Verify that setting RAG_PIPELINE_MODE='fast' skips intent/expansion."""
    import asyncio

    async def run_test():
        with patch.dict(os.environ, {"RAG_PIPELINE_MODE": "fast", "GEMINI_API_KEY": "fake"}):
            with patch('src.pipeline.ChromaDocumentStore'), \
                 patch('src.pipeline.GoogleGenAITextEmbedder') as mock_embedder, \
                 patch('src.pipeline.ChromaEmbeddingRetriever') as mock_retriever, \
                 patch('src.pipeline.GoogleGenAIChatGenerator') as mock_llm:

                # Setup mocks
                mock_embedder.return_value.run.return_value = {"embedding": [0.1]}
                mock_retriever.return_value.run.return_value = {"documents": []}
                mock_llm.return_value.run.return_value = {"replies": [MagicMock(text="Fast Answer")]}

                pipeline = PortfolioRagPipeline()
                result = await pipeline.run("Test Question")

                # Verify result
                assert result["intent"] == "fast_rag"
                assert result["answer"] == "Fast Answer"

                # Verify calls
                # 1. Embedder should be called with raw question (no expansion)
                mock_embedder.return_value.run.assert_called_with(text="Test Question")

                # 2. LLM should be called ONLY ONCE (for the final answer)
                assert mock_llm.return_value.run.call_count == 1

    asyncio.run(run_test())
