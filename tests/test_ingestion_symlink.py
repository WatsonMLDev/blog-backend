
import os
import pytest
from unittest.mock import MagicMock
from src.ingestion import GitRepositoryIngester

@pytest.fixture
def temp_repo_structure(tmp_path):
    """Creates a temporary repository structure with a symlink vulnerability."""
    base_dir = tmp_path / "test_ingestion_security"
    base_dir.mkdir()

    secret_file = base_dir / "secret.txt"
    secret_file.write_text("SUPER_SECRET_PASSWORD")

    repo_dir = base_dir / "repo"
    repo_dir.mkdir()

    # Create a symlink in repo pointing to secret file
    symlink_file = repo_dir / "link_to_secret.txt"
    try:
        os.symlink(secret_file.resolve(), symlink_file)
    except OSError:
        pytest.skip("Symlinks not supported on this OS")

    return repo_dir, secret_file

def test_ingester_ignores_symlinks_outside_repo(temp_repo_structure):
    repo_dir, secret_file = temp_repo_structure

    # Initialize Ingester with mocked external dependencies
    ingester = GitRepositoryIngester()
    ingester.repo_path = repo_dir
    ingester.files_to_index = ["link_to_secret.txt"]
    ingester.directories_to_index = []
    ingester.allowed_extensions = {".txt"}

    # Mock document store and pipeline to avoid actual DB/API calls
    ingester.document_store = MagicMock()
    ingester.embedding_pipeline = MagicMock()

    # Run extraction
    documents = ingester.extract_documents()

    # Verify that no documents contain the secret password
    for doc in documents:
        assert "SUPER_SECRET_PASSWORD" not in doc.content, "Vulnerability: Secret file content was read via symlink!"

    # Also verify that the file was skipped (documents list might be empty or missing this specific file)
    # Since we only requested this one file, documents should be empty
    assert len(documents) == 0, "Should not have extracted any documents from the malicious symlink."
