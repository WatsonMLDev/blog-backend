
import os
import pytest
from unittest.mock import MagicMock, patch
from src.ingestion import GitRepositoryIngester
import logging

class TestIngestionSecurity:
    @pytest.fixture
    def mock_env(self):
        with patch.dict(os.environ, {
            "REPO_URL": "https://github.com/test/repo.git",
            "GITHUB_TOKEN": "ghp_SECRET_TOKEN_123",
            "GEMINI_API_KEY": "fake_key"
        }):
            yield

    @pytest.fixture
    def ingester(self, mock_env):
        # We need to mock these to prevent actual initialization of Chroma/Haystack
        with patch("src.ingestion.GitRepositoryIngester._initialize_document_store"), \
             patch("src.ingestion.GitRepositoryIngester._initialize_pipeline"):
            ingester = GitRepositoryIngester()
            # Ensure attributes are set correctly for the test
            ingester.repo_url = "https://github.com/test/repo.git"
            ingester.github_token = "ghp_SECRET_TOKEN_123"
            # Mock repo_path to control exists() behavior
            ingester.repo_path = MagicMock()
            return ingester

    def test_github_token_not_leaked_in_logs_clone(self, ingester, caplog):
        """
        Verifies that GITHUB_TOKEN is NOT logged during clone.
        """
        caplog.set_level(logging.INFO)

        # Simulate repo not existing -> Clone path
        ingester.repo_path.exists.return_value = False

        with patch("subprocess.run"):
            try:
                ingester.clone_or_pull_repo()
            except Exception:
                pass

        # Verify fix
        log_text = caplog.text
        assert "ghp_SECRET_TOKEN_123" not in log_text, "GITHUB_TOKEN leaked in logs during clone!"
        assert "oauth2:***@" in log_text, "Masked token not found in logs."

    def test_github_token_not_leaked_in_logs_pull(self, ingester, caplog):
        """
        Verifies that GITHUB_TOKEN is NOT logged during pull.
        """
        caplog.set_level(logging.INFO)

        # Simulate repo exists -> Pull path
        ingester.repo_path.exists.return_value = True
        (ingester.repo_path / ".git").exists.return_value = True

        with patch("subprocess.run"):
             try:
                ingester.clone_or_pull_repo()
             except Exception:
                pass

        log_text = caplog.text
        assert "ghp_SECRET_TOKEN_123" not in log_text, "GITHUB_TOKEN leaked in logs during pull!"
        assert "oauth2:***@" in log_text, "Masked token not found in logs."
