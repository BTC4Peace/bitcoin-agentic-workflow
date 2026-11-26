"""Hardened stdlib HTTPS JSON client with retries and redirects disabled."""

from __future__ import annotations

import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

HTTP_TIMEOUT_SECONDS = 180
MAX_API_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_RETRY_AFTER_SECONDS = 30


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects so API credentials are never forwarded cross-host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise urllib.error.HTTPError(req.full_url, code, f"Redirect refused ({msg})", headers, fp)


_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _retry_delay(exc: urllib.error.HTTPError, attempt: int) -> int:
    default_wait = attempt * 5
    if exc.code == 429 and exc.headers:
        retry_after = exc.headers.get("Retry-After", "").strip()
        if retry_after.isdigit():
            return max(1, min(int(retry_after), MAX_RETRY_AFTER_SECONDS))
    return default_wait


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    provider_label: str = "API",
    timeout: int = HTTP_TIMEOUT_SECONDS,
    max_response_bytes: int = MAX_API_RESPONSE_BYTES,
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"Refusing non-HTTPS endpoint URL for {provider_label}.")

    data = json.dumps(payload).encode("utf-8")
    last_err = ""
    for attempt in range(1, 4):
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                raw = resp.read(max_response_bytes + 1)
                if len(raw) > max_response_bytes:
                    raise RuntimeError(f"{provider_label} response exceeded size limit.")
                decoded = json.loads(raw.decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise RuntimeError(
                        f"{provider_label} returned non-object JSON ({type(decoded).__name__})."
                    )
                return decoded
        except urllib.error.HTTPError as exc:
            body = (
                exc.read(500).decode("utf-8", errors="replace")
                if getattr(exc, "fp", None) is not None
                else str(exc.reason)
            )
            last_err = f"HTTP {exc.code}: {body}"
            if exc.code in (429, 500, 502, 503, 504) and attempt < 3:
                wait = _retry_delay(exc, attempt)
                print(
                    f"{provider_label} returned HTTP {exc.code} (attempt {attempt}/3); retrying in {wait}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(last_err) from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            http.client.HTTPException,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            last_err = str(exc)
            if attempt < 3:
                wait = attempt * 5
                print(
                    f"{provider_label} network error (attempt {attempt}/3): {exc}; retrying in {wait}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(last_err) from exc
    raise RuntimeError(last_err)
