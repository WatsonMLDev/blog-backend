## 2024-05-23 - [Blocking Calls in Async FastAPI Endpoints]
**Learning:** Even if a FastAPI endpoint is `async def`, calling synchronous blocking functions (like `time.sleep` or synchronous API clients) directly inside it will block the entire event loop, preventing concurrent request handling. This negates the benefit of `async`.
**Action:** When using synchronous libraries (like standard Haystack components or `google-genai` sync client) within an async application, always offload them to a thread pool using `loop.run_in_executor(None, func)`.
