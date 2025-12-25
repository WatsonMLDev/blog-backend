## 2024-12-25 - Asyncio Blocking
**Learning:** `async def` does not automatically make code non-blocking. Synchronous API calls (like `llm.run` or `time.sleep`) inside an `async def` function will still block the entire event loop, defeating the purpose of concurrency.
**Action:** Always wrap synchronous blocking calls in `await asyncio.to_thread(...)` (Python 3.9+) or `loop.run_in_executor(...)` when working within an async context.
