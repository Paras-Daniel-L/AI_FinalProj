"""
Chat client for an OpenAI-compatible API — OpenRouter by default. Only needs
`requests` (already a dependency), so no new packages.

What it does beyond a bare POST:

* THINKING OFF. For OpenRouter every request carries
  {"reasoning": {"enabled": false}}. Reasoning tokens are billed as output
  tokens, so leaving a reasoning model's thinking on can multiply the cost of
  a call many times over. ("exclude" would NOT do: excluded reasoning is
  still generated and billed — only "enabled": false prevents it.)
  Requests also carry {"provider": {"require_parameters": true}} so OpenRouter
  only routes to hosts that actually honor that setting instead of silently
  ignoring it. If a model refuses to run without reasoning, OpenRouter answers
  HTTP 400 and this client raises a clear error rather than paying for it.
* A HARD OUTPUT CAP (`max_tokens`) on every call, so no single call can run up
  a bill.
* RETRIES with backoff on rate limits (429), server errors (5xx), timeouts and
  network errors; it honors a Retry-After header. Everything else (bad key,
  out of credits, bad request) fails immediately with a readable message.
* USAGE TRACKING. Every call logs its tokens (and cost when OpenRouter reports
  it) and adds to running totals — see usage_summary(). If a response shows
  reasoning tokens anyway, a loud one-time warning is printed for that model,
  because that means thinking is NOT off and you are paying for it.

Settings (env vars; only the key is required):
    OPENROUTER_API_KEY   your key (LLM_API_KEY is accepted as an alias)
    LLM_BASE_URL         default https://openrouter.ai/api/v1
    LLM_DISABLE_THINKING default 1
    LLM_TIMEOUT          seconds per request, default 90
    LLM_MAX_RETRIES      retries after the first try, default 3
    LLM_EXTRA_BODY       JSON object merged into every request. When set it
                         REPLACES the built-in reasoning/provider fields — use
                         it for a non-OpenRouter endpoint that needs a
                         different "thinking off" switch.
"""

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL: str = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
REQUEST_TIMEOUT: float = float(os.environ.get("LLM_TIMEOUT", "90"))
MAX_RETRIES: int = int(os.environ.get("LLM_MAX_RETRIES", "3"))
DISABLE_THINKING: bool = os.environ.get("LLM_DISABLE_THINKING", "1").strip() not in ("0", "false", "False", "")
_IS_OPENROUTER: bool = "openrouter.ai" in BASE_URL


class LLMError(RuntimeError):
    """A model call failed in a way retrying won't fix (or retries ran out)."""


@dataclass
class ChatResult:
    text: str
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost: Optional[float] = None

    @property
    def truncated(self) -> bool:
        """True when the reply was cut off by the max_tokens cap."""
        return self.finish_reason == "length"


# ── Usage tracking ───────────────────────────────────────────────────────

_usage_lock = threading.Lock()
_usage: Dict[str, Dict[str, float]] = {}
_warned_reasoning: set = set()


def _record(model: str, r: ChatResult) -> None:
    with _usage_lock:
        u = _usage.setdefault(
            model,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cost": 0.0},
        )
        u["calls"] += 1
        u["prompt_tokens"] += r.prompt_tokens
        u["completion_tokens"] += r.completion_tokens
        u["reasoning_tokens"] += r.reasoning_tokens
        u["cost"] += r.cost or 0.0


def usage_summary() -> Dict[str, Dict[str, float]]:
    """Per-model totals since the process started (or reset_usage())."""
    with _usage_lock:
        return {m: dict(u) for m, u in _usage.items()}


def reset_usage() -> None:
    with _usage_lock:
        _usage.clear()
        _warned_reasoning.clear()


# ── Request building ─────────────────────────────────────────────────────

def _api_key() -> str:
    key = (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY") or "").strip()
    if not key:
        raise LLMError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file "
            "(OPENROUTER_API_KEY=sk-or-...), then restart the server."
        )
    return key


def _extra_body(provider_order: Optional[List[str]]) -> Dict[str, Any]:
    raw = os.environ.get("LLM_EXTRA_BODY", "").strip()
    if raw:
        try:
            body = json.loads(raw)
        except ValueError as e:
            raise LLMError(f"LLM_EXTRA_BODY is not valid JSON: {e}")
        if not isinstance(body, dict):
            raise LLMError("LLM_EXTRA_BODY must be a JSON object.")
        return body

    body: Dict[str, Any] = {}
    if _IS_OPENROUTER:
        if DISABLE_THINKING:
            body["reasoning"] = {"enabled": False}
            body["provider"] = {"require_parameters": True}
        if provider_order:
            body.setdefault("provider", {}).update(
                {"order": list(provider_order), "allow_fallbacks": True}
            )
    return body


def _error_message(body: Any, status: int) -> str:
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        msg = err.get("message") or err
        return f"HTTP {status}: {msg}"
    if err:
        return f"HTTP {status}: {err}"
    return f"HTTP {status}"


def _sleep_for(attempt: int, resp: Optional[requests.Response]) -> None:
    delay = min(2 ** attempt, 20)
    if resp is not None:
        try:
            delay = min(max(float(resp.headers.get("Retry-After", delay)), 0.0), 30.0)
        except (TypeError, ValueError):
            pass
    time.sleep(delay)


def chat(
    system: str,
    user: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    provider_order: Optional[List[str]] = None,
    role: str = "llm",
) -> ChatResult:
    """One chat completion. Raises LLMError on failure."""
    key = _api_key()
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload.update(_extra_body(provider_order))
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-Title": "Sagot AI",
    }
    url = f"{BASE_URL}/chat/completions"

    last_error = "unknown error"
    for attempt in range(MAX_RETRIES + 1):
        resp: Optional[requests.Response] = None
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = f"network error: {e}"
            if attempt < MAX_RETRIES:
                _sleep_for(attempt, None)
            continue

        status = resp.status_code
        try:
            body = resp.json()
        except ValueError:
            body = {}

        if status in (401, 403):
            raise LLMError(f"The API key was rejected ({_error_message(body, status)}). Check OPENROUTER_API_KEY in .env.")
        if status == 402:
            raise LLMError(f"Out of credits ({_error_message(body, status)}). Add credits on openrouter.ai.")
        if status == 429 or status == 408 or status >= 500:
            last_error = _error_message(body, status)
            if attempt < MAX_RETRIES:
                _sleep_for(attempt, resp)
            continue
        if status >= 400:
            hint = ""
            if status == 400 and "reasoning" in json.dumps(body).lower():
                hint = (" — this model may not allow thinking to be turned off; "
                        "choose another model or set LLM_DISABLE_THINKING=0 (costly)")
            raise LLMError(f"{_error_message(body, status)} (model {model}){hint}")

        # HTTP 200. OpenRouter can still report a provider failure in the body.
        if not isinstance(body, dict) or not body.get("choices"):
            err = body.get("error") if isinstance(body, dict) else None
            code = err.get("code") if isinstance(err, dict) else None
            if isinstance(code, int) and (code in (408, 429) or code >= 500):
                last_error = _error_message(body, code)
                if attempt < MAX_RETRIES:
                    _sleep_for(attempt, None)
                continue
            raise LLMError(f"Unexpected response from the model API: {_error_message(body, status)} (model {model})")

        return _parse(body, model, role)

    raise LLMError(f"Model call failed after {MAX_RETRIES + 1} attempts: {last_error} (model {model})")


def _parse(body: Dict[str, Any], model: str, role: str) -> ChatResult:
    choice = body["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):  # some providers return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))

    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    result = ChatResult(
        text=content,
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
        cost=usage.get("cost") if isinstance(usage.get("cost"), (int, float)) else None,
    )
    _record(model, result)

    line = f"💸 [LLM:{role}] {model} in={result.prompt_tokens} out={result.completion_tokens}"
    if result.reasoning_tokens:
        line += f" reasoning={result.reasoning_tokens}"
    if result.cost is not None:
        line += f" cost=${result.cost:.5f}"
    print(line)

    thinking_seen = result.reasoning_tokens > 0 or bool(message.get("reasoning"))
    if DISABLE_THINKING and thinking_seen and model not in _warned_reasoning:
        _warned_reasoning.add(model)
        print(
            f"\n🚨 [LLM] '{model}' returned reasoning even though thinking was turned off — "
            "you are being billed for it. Pick a provider that honors it "
            "(GENERATOR_PROVIDERS / VERIFIER_PROVIDERS) or another model.\n"
        )
    if result.truncated:
        print(f"⚠️  [LLM:{role}] {model} hit the max_tokens cap — the reply was cut off.")
    return result
