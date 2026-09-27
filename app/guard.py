"""
Public-sharing guard (rate limit, daily cap, admin lock).

Needed once the app is exposed through a tunnel (Cloudflare Tunnel / ngrok):

1. /upload and /reset are ADMIN routes. Anyone with the public link could
   otherwise wipe the vector store or add files. They only work when the
   request comes straight from this computer (no proxy/tunnel headers), i.e.
   from http://localhost:8000 in the owner's own browser. Tunnels forward
   requests from 127.0.0.1, so the client address alone is not enough: a
   request carrying tunnel headers is treated as public.
2. /query is limited per visitor IP per minute, and there is one global daily
   cap, so a shared link cannot drain the OpenRouter/Jina credits.

Env vars (all optional):
    RATE_LIMIT_PER_MIN   queries per IP per minute      (default 6)
    DAILY_QUERY_CAP      queries per day, all visitors  (default 400)
    ADMIN_LOCAL_ONLY     1 = lock /upload,/reset to this computer (default 1)
State is in memory: it resets when the server restarts.
"""

import os
import time
from collections import defaultdict, deque
from datetime import date

from fastapi import Request
from fastapi.responses import JSONResponse

RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "6"))
DAILY_QUERY_CAP = int(os.environ.get("DAILY_QUERY_CAP", "400"))
ADMIN_LOCAL_ONLY = os.environ.get("ADMIN_LOCAL_ONLY", "1").strip() not in ("0", "false", "False", "no")

ADMIN_PATHS = ("/upload", "/reset")
_PROXY_HEADERS = (
    "cf-connecting-ip", "x-forwarded-for", "x-real-ip", "forwarded", "cf-ray",
    "x-forwarded-host", "cf-visitor",
)

_hits = defaultdict(deque)  # ip -> timestamps of recent /query calls
_day = {"date": date.today(), "count": 0}


def client_ip(request: Request) -> str:
    h = request.headers
    ip = h.get("cf-connecting-ip") or (h.get("x-forwarded-for") or "").split(",")[0].strip()
    return ip or (request.client.host if request.client else "unknown")


def is_public(request: Request) -> bool:
    """True if the request came through a tunnel/proxy (or from another machine)."""
    if any(name in request.headers for name in _PROXY_HEADERS):
        return True
    host = request.client.host if request.client else ""
    return host not in ("127.0.0.1", "::1", "localhost")


def _blocked(status: int, message: str, retry_after: int = 0) -> JSONResponse:
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return JSONResponse({"detail": message}, status_code=status, headers=headers)


async def guard_middleware(request: Request, call_next):
    path = request.url.path

    if ADMIN_LOCAL_ONLY and path in ADMIN_PATHS and is_public(request):
        return _blocked(403, "This action is only available on the host computer.")

    if path == "/query" and request.method == "POST":
        today = date.today()
        if _day["date"] != today:
            _day["date"], _day["count"] = today, 0
        if _day["count"] >= DAILY_QUERY_CAP:
            return _blocked(429, "The daily question limit for this demo has been reached. Please try again tomorrow.")

        ip, now = client_ip(request), time.time()
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            wait = max(1, int(60 - (now - q[0])))
            return _blocked(429, f"Too many questions. Please wait {wait} seconds and try again.", wait)
        q.append(now)
        _day["count"] += 1

    return await call_next(request)
