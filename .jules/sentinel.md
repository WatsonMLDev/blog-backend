## 2024-05-23 - [CRITICAL] GitHub Token Leak in Logs
**Vulnerability:** The application was logging the full authenticated repository URL, including the `GITHUB_TOKEN`, during `git clone` and `git pull` operations.
**Learning:** Constructing URLs with embedded credentials for subprocess calls is a valid pattern, but using the same variable for logging creates a severe security risk. Developers often reuse variables for convenience, overlooking the side effect of logging secrets.
**Prevention:**
1. Always mask credentials in URLs before logging.
2. Use helper functions (like `_mask_url`) to centralize sanitization logic.
3. Add security regression tests that specifically assert the *absence* of secrets in logs.
