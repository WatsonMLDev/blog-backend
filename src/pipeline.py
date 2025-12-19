import os
import time
import logging
from typing import List, Dict, Any, Optional

# Haystack Core
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack.utils import Secret

# Haystack Integrations
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.embedders.google_genai import GoogleGenAITextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

logger = logging.getLogger(__name__)

class PortfolioRagPipeline:
    """
    A simplified, multi-stage Haystack RAG pipeline for a personal portfolio assistant,
    now with enhanced, detailed system prompts.
    """
    def __init__(self):
        """Initializes configuration, components, and prompt templates."""
        self._load_config()
        self._initialize_components()
        self._initialize_prompts()
        logger.info("Portfolio RAG Pipeline initialized with enhanced prompts.")

    def _load_config(self):
        """Loads configuration from environment variables."""
        self.top_k = int(os.environ.get("RAG_TOP_K", "5"))
        self.gemini_key = Secret.from_env_var("GEMINI_API_KEY")
        self.doc_store_path = os.environ.get("DOCUMENT_STORE_PATH", "./data/chroma_db")
        
        # Model selection for different pipeline stages
        self.intent_model = os.environ.get("INTENT_MODEL", "gemini-3-flash-preview")
        self.expander_model = os.environ.get("EXPANDER_MODEL", "gemini-3-flash-preview")
        self.chat_model = os.environ.get("CHAT_MODEL", "gemini-3-flash-preview")
        self.rag_model = os.environ.get("RAG_MODEL", "gemini-3-flash-preview")  # Use experimental for final answers

    def _initialize_components(self):
        """Initializes core Haystack components."""
        self.document_store = ChromaDocumentStore(
            collection_name="git_repo_docs",
            persist_path=self.doc_store_path
        )
        self.query_embedder = GoogleGenAITextEmbedder(api_key=self.gemini_key)
        self.retriever = ChromaEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.top_k
        )
        
        # Separate LLM instances for different pipeline stages
        self.intent_llm = GoogleGenAIChatGenerator(model=self.intent_model, api_key=self.gemini_key)
        self.expander_llm = GoogleGenAIChatGenerator(model=self.expander_model, api_key=self.gemini_key)
        self.chat_llm = GoogleGenAIChatGenerator(model=self.chat_model, api_key=self.gemini_key)
        self.rag_llm = GoogleGenAIChatGenerator(model=self.rag_model, api_key=self.gemini_key)

    def _initialize_prompts(self):
        """Defines and builds all necessary prompt templates with detailed instructions."""
        # --- Intent Classification Prompt ---
        self.intent_prompt_builder = ChatPromptBuilder(template=[
            ChatMessage.from_user(
                "You are an expert at classifying user intent. Based on the conversation history and the user's message, "
                "is their intent to 'search' for portfolio information, or just to 'chat'?\n\n"
                "IMPORTANT RULES:\n"
                "1. If the user is responding to a question from the assistant (like 'yes', 'tell me more', 'please', 'elaborate'), classify as 'search'\n"
                "2. If the user is asking about portfolio content, projects, or experience, classify as 'search'\n"
                "3. Only classify as 'chat' if the user is making small talk unrelated to the portfolio (like 'how are you', 'what's the weather')\n\n"
                "Answer only with the word 'search' or 'chat'.\n\n"
                "Conversation History:\n{{chat_history}}\n\n"
                "User message: {{question}}"
            )],
            required_variables=["question", "chat_history"]
            )
        
        # --- Conversational Chat Prompt (Enhanced) ---
        self.chat_prompt_builder = ChatPromptBuilder(template=[
            ChatMessage.from_user(
                "You are a friendly portfolio assistant. The user has said something conversational.\n"
                "Acknowledge their message and gently guide them back to your purpose of providing portfolio information.\n"
                "For example: 'Thanks for sharing! My main purpose is to help you learn about this portfolio. Is there something specific you'd like to know?'\n\n"
                "User message: {{question}}"
            )],
            required_variables=["question"]
        )

        # --- Query Expansion Prompt (Enhanced) ---
        self.expander_prompt_builder = ChatPromptBuilder(template=[
            ChatMessage.from_user(
                "Given the conversation history and the user's latest question, rewrite the question "
                "into a concise, self-contained search query optimized for embedding-based retrieval.\n\n"
                "IMPORTANT: If the user's question is a follow-up (like 'yes', 'tell me more', 'can you elaborate'), "
                "expand it based on the previous conversation to include what topic they want more information about.\n\n"
                "For example:\n"
                "- If assistant asked 'Would you like more details about project X?' and user said 'yes', "
                "  expand to: 'Tell me more details about project X'\n"
                "- If the question is already clear and specific, return it as is.\n\n"
                "Conversation History:\n{{chat_history}}\n\n"
                "User's Question: {{question}}\n\n"
                "Expanded Search Query:"
            )],
            required_variables=["question", "chat_history"]
        )

        # --- Final Answer RAG Prompt (Enhanced) ---
        rag_prompt_template = """
        You are a friendly and professional assistant for Charlie Watson's portfolio website.

        ## Style and Tone:
        - **Be Succinct:** Your main goal is to be concise. Keep answers to the point.
        - Start with a friendly, natural opening.
        - End with an open-ended question that invites the user to ask for more, like "Would you like a more detailed explanation?"

        ## Your Task:
        1.  **Provide a brief summary** that answers the user's question based *only* on the "Relevant Context" provided below.
        2.  **Limit your main answer to 3-4 sentences.** Give a high-level overview, not a detailed explanation.
        3.  If the context does **not** contain the information to answer the question, respond *only* with: "Sorry, I couldn't find any information about that in the portfolio."

        ---
        User's Original Question: {{question}}
        My Understanding of the Question (for search): {{expanded_query}}
        ---
        Relevant Context:
        {% for doc in documents %}
        --- Document Source: {{ doc.meta.get('file_name', 'Unknown') }} ---
        {{ doc.content }}
        {% endfor %}
        ---

        Answer:
        """
        self.rag_prompt_builder = ChatPromptBuilder(
            template=[ChatMessage.from_user(rag_prompt_template)],
            required_variables=["question", "expanded_query", "documents"]
        )

    def _run_llm(self, prompt_builder: ChatPromptBuilder, data: Dict[str, Any], llm: GoogleGenAIChatGenerator) -> str:
        """Helper function to run a prompt through a specific LLM and get the text reply."""
        prompt = prompt_builder.run(**data).get("prompt", [])
        response = llm.run(messages=prompt)
        return response["replies"][0].text if response.get("replies") else ""

    async def run(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Runs the multi-stage RAG pipeline.
        Returns a dictionary with the final response and retrieved documents.
        """
        start_time = time.time()
        chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in (chat_history or [])])
        logger.info(f"Running pipeline for question: {question}")
        
        # 1. Classify Intent (with chat history for context)
        intent = self._run_llm(self.intent_prompt_builder, {"question": question, "chat_history": chat_history_str}, self.intent_llm).strip().lower()
        logger.info(f"Classified intent: '{intent}'")
        
        # 2a. Handle "chat" intent
        if "chat" in intent:
            logger.info("Handling as a conversational chat.")
            final_response = self._run_llm(self.chat_prompt_builder, {"question": question}, self.chat_llm)
            latency = time.time() - start_time
            return {
                "answer": final_response,
                "documents": [],
                "intent": "chat",
                "latency": latency,
                "model": self.chat_model
            }

        # 2b. Handle "search" intent
        logger.info("Handling as a search query.")
        
        # 3. Expand Query
        expanded_query = self._run_llm(self.expander_prompt_builder, {"question": question, "chat_history": chat_history_str}, self.expander_llm)
        logger.info(f"Expanded query: '{expanded_query}'")
        
        # 4. Retrieve Documents
        query_embedding = self.query_embedder.run(text=expanded_query).get("embedding", [])
        documents = self.retriever.run(query_embedding=query_embedding).get("documents", [])
        logger.info(f"Retrieved {len(documents)} documents.")
        
        # 5. Generate Final Answer (Note: No documents is handled by the prompt now)
        final_response = self._run_llm(
            self.rag_prompt_builder,
            {
                "question": question,
                "expanded_query": expanded_query,
                "documents": documents
            },
            self.rag_llm
        )
        logger.info("Generated final answer.")
        
        latency = time.time() - start_time
        return {
            "answer": final_response,
            "documents": documents,
            "intent": "search",
            "latency": latency,
            "model": self.rag_model
        }
