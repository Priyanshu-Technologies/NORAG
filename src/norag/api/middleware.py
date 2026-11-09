import logging
import threading
import time
from collections import defaultdict, deque
from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("norag.api")


class RateLimitMiddleware:
    """Fixed-window per-client request limiter using in-process state."""

    def __init__(self, app: ASGIApp, limit_per_minute: int) -> None:
        if limit_per_minute < 1:
            raise ValueError("limit_per_minute must be positive")

        self.app = app
        self.limit = limit_per_minute
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        key = client[0] if client else "unknown"
        now = time.monotonic()

        with self._lock:
            window = self._windows[key]

            while window and now - window[0] >= 60.0:
                window.popleft()

            if len(window) >= self.limit:
                response = JSONResponse(
                    {"detail": "Rate limit exceeded. Try again shortly."},
                    status_code=429,
                    headers={"Retry-After": "60"},
                )
                await response(scope, receive, send)
                return

            window.append(now)

        await self.app(scope, receive, send)


class RequestLoggingMiddleware:
    """Attach a request id and log method, path, status and duration."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid4())
        state = scope.setdefault("state", {})
        state["request_id"] = request_id

        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message) -> None:
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers.append("X-Request-ID", request_id)

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "%s %s -> %s in %.1fms [%s]",
                scope.get("method", ""),
                scope.get("path", ""),
                status_code,
                duration_ms,
                request_id,
            )
