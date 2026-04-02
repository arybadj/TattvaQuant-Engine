import time
from collections import defaultdict, deque

from fastapi import Request
from redis.asyncio import from_url
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import get_settings

settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._memory_buckets: dict[str, deque[float]] = defaultdict(deque)
        self._redis = from_url(settings.redis_url, decode_responses=True)

    async def dispatch(self, request: Request, call_next):
        key = request.client.host if request.client else "unknown"
        if not await self._allow_request(key):
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        return await call_next(request)

    async def _allow_request(self, key: str) -> bool:
        redis_key = f"rate-limit:{key}:{int(time.time() // 60)}"
        try:
            count = await self._redis.incr(redis_key)
            if count == 1:
                await self._redis.expire(redis_key, 60)
            return count <= settings.rate_limit_per_minute
        except Exception:
            now = time.time()
            bucket = self._memory_buckets[key]
            while bucket and now - bucket[0] > 60:
                bucket.popleft()
            if len(bucket) >= settings.rate_limit_per_minute:
                return False
            bucket.append(now)
            return True

