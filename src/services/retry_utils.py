import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from google.genai import errors

logger = logging.getLogger(__name__)

def is_rate_limit_error(exception):
    """Check if the exception is a 429 Resource Exhausted error."""
    if isinstance(exception, errors.ClientError):
        # Check if the error message or code indicates 429
        # Based on the error log: google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED
        # The exception string representation usually starts with the code.
        return "429" in str(exception) or "RESOURCE_EXHAUSTED" in str(exception)
    return False

# Decorator for retrying with exponential backoff
# Waits 1s, 2s, 4s, ... up to 60s, with random jitter.
retry_with_backoff = retry(
    retry=retry_if_exception_type(errors.ClientError), # We can refine this to only retry 429 if needed, but for now ClientError is safe-ish if we check inside or just retry all client errors that might be transient? 
    # Actually, let's use a custom predicate to be safe and only retry 429s.
    # But wait, tenacity's retry_if_exception_type takes a class.
    # Let's use retry_if_exception to use our custom function.
    
    # Update: The error seen is `google.genai.errors.ClientError`.
    # Let's retry on that specific error if it matches our check.
    retry=retry_if_exception(is_rate_limit_error),
    
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10), # Try up to 10 times. With exp backoff this covers a few minutes.
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
