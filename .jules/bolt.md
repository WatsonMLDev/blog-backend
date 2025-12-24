## 2024-05-23 - Blocking Calls in Async Pipeline
**Learning:** Even if a method is defined as `async`, calling synchronous blocking methods (like `llm.run` or network I/O) inside it without `await`ing them on an executor will block the entire asyncio event loop. This serializes concurrent requests, degrading performance significantly.
**Action:** Always wrap synchronous I/O or CPU-bound operations in `loop.run_in_executor` within async functions to ensure true concurrency.
