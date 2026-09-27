"""
Public-sharing guard (rate limit, daily cap, admin lock).

Needed once the app is exposed through a tunnel (ngrok / Cloudflare Tunnel):

1. /upload and /reset are ADMIN routes. Anyone with the public link could
   otherwise wipe the vector store or add files. They only work when the
   request comes straight from this computer (no proxy/tunnel headers), i.e.
   from http://localhost:8000 in the owner's own browser, Swagger (/docs) or
   curl. Tunnels forward requests from 127.0.0.1, so the client address alone
   is not enough: a request carrying tunnel headers is treated as public.

   Two browser-based tricks are also blocked, because a web page open in the
   owner's own browser runs ON this computer:
   - Cross-site requests: another site's page sending a form/fetch to
     http://localhost:8000/upload. Browsers label these with an Origin header
     and Sec-Fetch-Site; anything not from this app's own page is refused.
     (Removing CORS is not enough on its own: a multipart form POST is a
     "simple request" that browsers send without asking first.)
   - DNS rebinding: an attacker's domain made to point at 127.0.0.1, so the
     request looks same-origin. Admin routes therefore also require the Host
     header to be localhost / 127.0.0.1.

2. /query is limited per visitor IP per minute, and there is one global daily
   cap, so a shared link cannot drain the OpenRouter/Jina credits.

Visitor IP: the tunnel appends the address it actually saw to the END of
X-Forwarded-For. Anything before it was written by the visitor and can be
faked, so the LAST entry is used. CF-Connecting-IP is only trusted when
TRUST_CF_CONNECTING_IP=1 (i.e. you really run behind Cloudflare) — through
ngrok a visitor could send that header themselves.

Env vars (all optional):
    RATE_LIMIT_PER_MIN      queries per IP per minute           (default 6)
    DAILY_QUERY_CAP         queries per day, all visitors       (default 100)
    ADMIN_LOCAL_ONLY        1 = lock /upload,/reset to this computer (default 1)
    TRUST_CF_CONNECTING_IP  1 = use Cloudflare's CF-Connecting-IP (default 0)
State is in memory: it resets when the server restarts.

Budget note: the worst case is about 1.1 US cents per question (3 attempts,
every token cap hit), so DAILY_QUERY_CAP=100 bounds a day at roughly $1.10.
"""

import os
import time
from collections import defaultdict, deque
from datetime import date
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import Request
from fastapi.responses import JSONResponse

load_dotenv()  # read .env here too, instead of relying on another module loading it first


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "")


RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "6"))
DAILY_QUERY_CAP = int(os.environ.get("DAILY_QUERY_CAP", "100"))
ADMIN_LOCAL_ONLY = _flag("ADMIN_LOCAL_ONLY", "1")
TRUST_CF_CONNECTING_IP = _flag("TRUST_CF_CONNECTING_IP", "0")

ADMIN_PATHS = ("/upload", "/reset")
_PROXY_HEADERS = (
    "cf-connecting-ip", "x-forwarded-for", "x-real-ip", "forwarded", "cf-ray",
    "x-forwarded-host", "cf-visitor",
)
_LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost")

_hits = defaultdict(deque)  # ip -> timestamps of recent /query calls
_day = {"date": date.today(), "count": 0}
_last_prune = {"t": 0.0}


def client_ip(request: Request) -> str:
    """The visitor's address as the tunnel saw it (not whatever the visitor claims)."""
    h = request.headers
    if TRUST_CF_CONNECTING_IP and h.get("cf-connecting-ip"):
        return h["cf-connecting-ip"].strip()
    forwarded = [p.strip() for p in (h.get("x-forwarded-for") or "").split(",") if p.strip()]
    if forwarded:
        return forwarded[-1]
    return request.client.host if request.client else "unknown"


def is_public(request: Request) -> bool:
    """True if the request came through a tunnel/proxy (or from another machine)."""
    if any(name in request.headers for name in _PROXY_HEADERS):
        return True
    host = request.client.host if request.client else ""
    return host not in _LOCAL_HOSTS


def _hostname(value: str) -> str:
    """'localhost:8000' -> 'localhost'; '[::1]:8000' -> '::1'."""
    try:
        return (urlsplit(f"//{value}").hostname or "").lower()
    except ValueError:
        return ""


def is_cross_site(request: Request) -> bool:
    """
    True if a browser sent this request from a page that is not this app
    running on localhost (another website, or a DNS-rebinding domain).
    curl and similar tools send none of these headers and are not affected.
    """
    h = request.headers
    if _hostname(h.get("host", "")) not in _LOCAL_HOSTS:
        return True  # DNS rebinding: the page's domain resolves to 127.0.0.1
    if h.get("sec-fetch-site", "").lower() in ("cross-site", "same-site"):
        return True
    origin = h.get("origin")
    if origin:
        if origin == "null":
            return True  # sandboxed iframe / file:// page
        if urlsplit(origin).netloc.lower() != h.get("host", "").lower():
            return True
    return False


def _prune(now: float) -> None:
    """Drop IPs with no queries in the last minute (at most once a minute)."""
    if now - _last_prune["t"] < 60:
        return
    _last_prune["t"] = now
    for ip in [ip for ip, q in _hits.items() if not q or now - q[-1] > 60]:
        del _hits[ip]


def _blocked(status: int, message: str, retry_after: int = 0) -> JSONResponse:
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return JSONResponse({"detail": message}, status_code=status, headers=headers)


async def guard_middleware(request: Request, call_next):
    path = request.url.path

    if ADMIN_LOCAL_ONLY and path in ADMIN_PATHS:
        if is_public(request) or is_cross_site(request):
            print(f"🛡️  [Guard] blocked admin request to {path} from {client_ip(request)}")
            return _blocked(403, "This action is only available on the host computer.")

    if path == "/query" and request.method == "POST":
        today = date.today()
        if _day["date"] != today:
            _day["date"], _day["count"] = today, 0
        if _day["count"] >= DAILY_QUERY_CAP:
            return _blocked(429, "The daily question limit for this demo has been reached. Please try again tomorrow.")

        ip, now = client_ip(request), time.time()
        _prune(now)
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            wait = max(1, int(60 - (now - q[0])))
            return _blocked(429, f"Too many questions. Please wait {wait} seconds and try again.", wait)
        q.append(now)
        _day["count"] += 1

    return await call_next(request)
