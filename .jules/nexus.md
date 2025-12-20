# NEXUS LOG - 2024-05-20

## 🎯 Logic Win: Latency & Intent Tracking

**Observation:**
The previous logging system tracked "message_sent" but missed critical performance metrics like how long the LLM took to respond, which model was used, and whether the user's intent was "chat" or "search".

**Implementation:**
- **Modified `src/pipeline.py`:** Added `time` tracking to measure end-to-end pipeline latency. The `run` method now returns a dictionary containing `latency`, `intent` (chat/search), and the `model` used for generation.
- **Modified `main.py`:** Updated the `chat_endpoint` to extract these metrics and log an `inference_completed` event to `stats_tracker`.

**Impact:**
- **Observability:** We can now analyze latency trends per model and verify if the intent classifier is performing as expected.
- **Optimization:** Identifying slow queries or intent misclassifications becomes data-driven.
- **Non-breaking:** The API response remains unchanged for the frontend.

# NEXUS LOG - 2024-05-21

## 🎯 Logic Win: Enhanced Query Expansion for Pronoun Resolution

**Observation:**
The RAG pipeline struggled with follow-up questions containing pronouns (e.g., "What does it do?"), as the generated search queries were often vague or missing context.

**Implementation:**
- **Modified `src/pipeline.py`:** Updated the `expander_prompt_builder` system prompt to explicitly instruct the model to resolve pronouns like "it", "that", and "his" using the conversation history.

**Impact:**
- **Accuracy:** The search queries generated for follow-up questions are now self-contained and much more likely to retrieve relevant documents.
- **DX:** Added `tests/test_expander_logic.py` (during dev) to ensure the prompt instructions persist.
- **Non-breaking:** Pure prompt engineering change; no code logic or API modifications.
