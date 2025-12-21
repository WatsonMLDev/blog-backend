
# NEXUS LOG - 2024-05-23: Optimize RAG System Prompt

## 🎯 Logic Win: RAG Prompt Optimization

**Observation:**
The previous RAG system prompt was somewhat generic and lacked explicit instructions for professional tone and formatting (e.g., bullet points).

**Implementation:**
- **Modified `src/pipeline.py`:** Updated `rag_system_msg` to:
    - Enforce a "professional portfolio assistant" persona.
    - Explicitly forbid using outside knowledge.
    - Standardize the "not found" fallback message.
    - Request bullet points for lists.
    - Limit answers to max 3 sentences for conciseness.

**Impact:**
- **Quality:** Answers should be more consistent, professional, and easier to read.
- **Token Efficiency:** Concise instructions may slightly reduce output token usage.
- **Non-breaking:** Strictly an internal prompt change.
