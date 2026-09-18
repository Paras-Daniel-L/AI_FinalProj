import json
import hashlib
import os
from typing import Optional, List, Dict, Any
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True,
    socket_timeout=2.0
)

def is_redis_available() -> bool:
    """Safely check if Redis instance is active."""
    try:
        return bool(redis_client.ping())
    except Exception:
        return False

def make_cache_key(prefix: str, query: str, filter_val: Optional[str] = None) -> str:
    """Generate a deterministic SHA-256 hash key for retrieval requests."""
    normalized = f"{query.strip().lower()}:{filter_val or 'all'}"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"