#!/usr/bin/env python3
"""
ingestion.py

Git repository content ingestion pipeline for loading documents into ChromaDB.
Follows modern Haystack architecture patterns with pipeline components.
"""

# Standard Library
import os
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

# Environment & Haystack
from dotenv import load_dotenv
from haystack import Pipeline, Document
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder

# Setup logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

class GitRepositoryIngester:
    """
    A pipeline to ingest files from a Git repository, create embeddings,
    and store them in a ChromaDB vector store.
    """

    def __init__(self):
        """Initializes the ingester by loading configuration and setting up Haystack components."""
        self._load_config()
        self.document_store = self._initialize_document_store()
        self.embedding_pipeline = self._initialize_pipeline()
        logger.info(f"GitRepositoryIngester initialized for repo: {self.repo_url}")

    def _load_config(self):
        """Loads configuration from environment variables."""
        self.repo_url = os.environ.get("REPO_URL", "REPO_URL_HERE")
        self.repo_path = Path(os.environ.get("REPO_PATH", "./temp_repo"))
        self.files_to_index = os.environ.get("FILES_TO_INDEX", "").split(",")
        self.directories_to_index = os.environ.get("DIRECTORIES_TO_INDEX", "").split(",")
        self.document_store_path = Path(os.environ.get("DOCUMENT_STORE_PATH", "./data/chroma_db"))
        self.gemini_api_key = Secret.from_env_var("GEMINI_API_KEY")
        self.github_token = os.environ.get("GITHUB_TOKEN")
        self.allowed_extensions = {".md", ".txt", ".py", ".ts", ".js", ".json", ".yaml", ".yml", ".html", ".htm", ".csv"}

    def _initialize_document_store(self) -> ChromaDocumentStore:
        """Sets up and returns the ChromaDocumentStore."""
        chroma_persist_dir = self.document_store_path
        chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ChromaDB persist path: {chroma_persist_dir}")
        return ChromaDocumentStore(
            collection_name="git_repo_docs",
            persist_path=str(self.document_store_path) # Use the path directly
        )

    def _initialize_pipeline(self) -> Pipeline:
        """Builds and returns the Haystack indexing pipeline."""
        pipeline = Pipeline()
        pipeline.add_component("embedder", GoogleGenAIDocumentEmbedder(api_key=self.gemini_api_key))
        pipeline.add_component("writer", DocumentWriter(document_store=self.document_store))
        pipeline.connect("embedder.documents", "writer.documents")
        return pipeline

    def clone_or_pull_repo(self) -> None:
        """Clones the repository or pulls the latest changes."""
        try:
            auth_repo_url = self.repo_url
            if self.github_token:
                # Insert token into URL for authentication
                auth_repo_url = self.repo_url.replace("https://", f"https://oauth2:{self.github_token}@")

            if self.repo_path.exists() and (self.repo_path / ".git").exists():
                logger.info(f"Repo exists. Updating remote URL and pulling latest changes from {auth_repo_url}...")
                # Update remote URL to include token auth
                subprocess.run(["git", "remote", "set-url", "origin", auth_repo_url], cwd=str(self.repo_path), check=True, text=True, capture_output=True)
                subprocess.run(["git", "pull"], cwd=str(self.repo_path), check=True, text=True, capture_output=True)
            else:
                if self.repo_path.exists():
                    shutil.rmtree(self.repo_path)
                logger.info(f"Cloning repository {auth_repo_url} into {self.repo_path}...")
                subprocess.run(["git", "clone", auth_repo_url, str(self.repo_path)], check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Git operation failed: {e.stderr}")
            raise RuntimeError("Failed to clone or pull repository.") from e

    def _gather_file_paths(self) -> List[Path]:
        """Gathers a unique set of file paths to be indexed."""
        paths = set()
        for rel_file in filter(None, [f.strip() for f in self.files_to_index]):
            candidate = self.repo_path / rel_file
            if candidate.is_file():
                paths.add(candidate)
        for rel_dir in filter(None, [d.strip() for d in self.directories_to_index]):
            base_dir = self.repo_path / rel_dir
            if base_dir.is_dir():
                for path in base_dir.rglob("*"):
                    if path.is_file() and path.suffix.lower() in self.allowed_extensions:
                        paths.add(path)
        return list(paths)

    def extract_documents(self) -> List[Document]:
        """Extracts content from files and converts them into Haystack Documents."""
        file_paths = self._gather_file_paths()
        documents = []
        for path in file_paths:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                doc = Document(content=content, meta={"source": str(path.relative_to(self.repo_path)), "file_name": path.name})
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Could not read file {path}, skipping. Error: {e}")
        logger.info(f"Extracted {len(documents)} documents from {len(file_paths)} files.")
        return documents

    def run(self, cleanup_repo: bool = True) -> bool:
        """
        Executes the full ingestion pipeline: clone/pull, extract, and index.
        Returns True on success, False on failure.
        """
        try:
            # Step 1: Clone or pull the repository
            self.clone_or_pull_repo()
            
            # Step 2: Extract content into Haystack Documents
            documents = self.extract_documents()
            if not documents:
                logger.warning("No documents found to ingest. Stopping.")
                return True

            # Step 3: Run the embedding and writing pipeline
            self.embedding_pipeline.run({"embedder": {"documents": documents}})
            logger.info("✅ Ingestion pipeline finished successfully.")
            return True
            
        except Exception as e:
            logger.exception(f"❌ Ingestion pipeline failed: {e}")
            return False
        finally:
            if cleanup_repo and self.repo_path.exists():
                logger.info(f"Cleaning up temporary repository at {self.repo_path}")
                shutil.rmtree(self.repo_path)

# --- Standalone Test Function ---

def test_document_embeddings(document_store: ChromaDocumentStore) -> None:
    """Verifies that documents in the document store have embeddings."""
    logger.info("🧪 Verifying document embeddings...")
    docs = document_store.filter_documents()
    if not docs:
        logger.error("Verification failed: No documents found in the store.")
        return

    docs_with_embeddings = [doc for doc in docs if doc.embedding is not None and len(doc.embedding) > 0]
    
    if len(docs_with_embeddings) == len(docs):
        dim = len(docs_with_embeddings[0].embedding)
        logger.info(f"✅ Verification successful: All {len(docs)} documents have embeddings (dim: {dim}).")
    else:
        logger.error(f"❌ Verification failed: {len(docs_with_embeddings)}/{len(docs)} documents have embeddings.")


# --- Main Execution ---

def main():
    ingester = GitRepositoryIngester()
    if ingester.repo_url == "REPO_URL_HERE" or not ingester.gemini_api_key.resolve_value():
        logger.error("Please set REPO_URL and GEMINI_API_KEY in your .env file.")
        return 1

    success = ingester.run()
    
    if success:
        # If ingestion was successful, run the verification test
        test_document_embeddings(ingester.document_store)
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())
