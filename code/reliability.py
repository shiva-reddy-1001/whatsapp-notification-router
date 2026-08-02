"""Bounded, configurable retries shared by remote and local model adapters."""
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar


T = TypeVar("T")


class InvalidModelResponse(RuntimeError):
    """A syntactically invalid model answer that can be repaired once retried."""


class RetryDeadlineExceeded(TimeoutError):
    pass


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int
    mode: str
    base_seconds: float
    max_seconds: float
    jitter_seconds: float
    deadline_seconds: float

    def delay(self, retry_number: int, random_value: float) -> float:
        if self.mode == "none":
            return 0.0
        base = self.base_seconds
        if self.mode == "exponential":
            base *= 2 ** max(0, retry_number - 1)
        return min(self.max_seconds, base + self.jitter_seconds * random_value)


def is_retryable(error: Exception) -> bool:
    """Classify errors without importing provider-specific exception classes."""
    if isinstance(error, InvalidModelResponse):
        return True
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    status = getattr(error, "status_code", None)
    if status == 429 or (isinstance(status, int) and 500 <= status < 600):
        return True
    name = type(error).__name__.lower()
    text = str(error).lower()
    transient = ("timeout", "connection", "temporarily unavailable", "rate limit",
                 "broken pipe", "connection reset")
    terminal = ("authentication", "permission", "unauthorized", "invalid model",
                "model not found", "not found")
    return not any(word in text or word in name for word in terminal) and any(
        word in text or word in name for word in transient)


def run_with_retry(operation: Callable[[int], T], policy: RetryPolicy,
                   sleeper: Callable[[float], None] = time.sleep,
                   clock: Callable[[], float] = time.monotonic,
                   random_source: Callable[[], float] = random.random,
                   on_retry: Optional[Callable[[int, Exception, float], None]] = None) -> T:
    started = clock()
    for attempt in range(policy.max_retries + 1):
        try:
            return operation(attempt)
        except Exception as error:
            if (policy.mode == "none" or attempt >= policy.max_retries or
                    not is_retryable(error)):
                raise
            delay = policy.delay(attempt + 1, random_source())
            elapsed = clock() - started
            if elapsed + delay >= policy.deadline_seconds:
                raise RetryDeadlineExceeded("retry deadline exhausted") from error
            if on_retry:
                on_retry(attempt + 1, error, delay)
            sleeper(delay)
    raise AssertionError("unreachable")
