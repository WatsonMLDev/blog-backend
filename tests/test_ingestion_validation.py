
import pytest
from unittest.mock import MagicMock, patch
from src.ingestion import GitRepositoryIngester

class TestIngestionValidation:
    @pytest.fixture
    def ingester(self):
        # Mock initialization to avoid external dependencies
        with patch("src.ingestion.GitRepositoryIngester._initialize_document_store"), \
             patch("src.ingestion.GitRepositoryIngester._initialize_pipeline"), \
             patch.dict("os.environ", {"REPO_URL": "https://valid.url", "GEMINI_API_KEY": "fake"}):
            ingester = GitRepositoryIngester()
            ingester.repo_path = MagicMock()
            ingester.repo_path.exists.return_value = False
            return ingester

    def test_validate_repo_url_valid_https(self, ingester):
        """Test that a valid HTTPS URL is accepted."""
        ingester.repo_url = "https://github.com/test/repo.git"
        with patch("subprocess.run") as mock_run:
            ingester.clone_or_pull_repo()
            # Should not raise exception and should call git clone
            assert mock_run.called

    def test_validate_repo_url_valid_ssh(self, ingester):
        """Test that a valid SSH URL is accepted."""
        ingester.repo_url = "ssh://git@github.com/test/repo.git"
        with patch("subprocess.run") as mock_run:
            ingester.clone_or_pull_repo()
            assert mock_run.called

    def test_validate_repo_url_valid_git_at(self, ingester):
        """Test that a valid git@ URL is accepted."""
        ingester.repo_url = "git@github.com:test/repo.git"
        with patch("subprocess.run") as mock_run:
            ingester.clone_or_pull_repo()
            assert mock_run.called

    def test_validate_repo_url_invalid_scheme_file(self, ingester):
        """Test that file:// scheme is rejected."""
        ingester.repo_url = "file:///etc/passwd"
        with pytest.raises(ValueError, match="Invalid repository URL scheme"):
            ingester.clone_or_pull_repo()

    def test_validate_repo_url_invalid_start_dash(self, ingester):
        """Test that URLs starting with dash are rejected (argument injection)."""
        ingester.repo_url = "-oProxyCommand=calc"
        with pytest.raises(ValueError, match="Invalid repository URL"):
            ingester.clone_or_pull_repo()

    def test_validate_repo_url_invalid_characters(self, ingester):
        """Test that URLs with newlines or other suspicious chars are rejected."""
        ingester.repo_url = "https://github.com/repo.git\nPayload"
        with pytest.raises(ValueError, match="Invalid repository URL"):
            ingester.clone_or_pull_repo()
