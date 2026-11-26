"""Native Google Gemini REST (:generateContent) provider adapter."""

from __future__ import annotations

import re
import sys
from typing import Any

from reviewer.http_client import post_json
from reviewer.providers.base import BaseProvider, TokenUsage, ToolCall, TurnResult

MODEL_RE = re.compile(r"^[a-zA-Z0-9._-]+$")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _to_gemini_schema_type(json_type: str) -> str:
    return json_type.upper()


def _to_gemini_tools(tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    declarations = []
    for spec in tool_specs:
        props = {}
        for prop_name, prop_def in spec["properties"].items():
            props[prop_name] = {
                "type": _to_gemini_schema_type(prop_def["type"]),
                "description": prop_def["description"],
            }
        params: dict[str, Any] = {
            "type": "OBJECT",
            "properties": props,
        }
        if spec.get("required"):
            params["required"] = spec["required"]
        declarations.append(
            {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": params,
            }
        )
    return [{"functionDeclarations": declarations}]


class GeminiProvider(BaseProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        tool_specs: list[dict[str, Any]],
    ) -> None:
        if not MODEL_RE.fullmatch(model):
            raise ValueError(f"Invalid Gemini model identifier: {model!r}")
        self._model = model
        self._api_url = f"{GEMINI_BASE_URL}/{model}:generateContent"
        self._api_key = api_key
        self._tools = _to_gemini_tools(tool_specs)
        self._system_prompt = ""
        self._contents: list[dict[str, Any]] = []

    def init_conversation(self, system_prompt: str, user_content: str) -> None:
        self._system_prompt = system_prompt
        self._contents = [
            {
                "role": "user",
                "parts": [{"text": user_content}],
            }
        ]

    def call_turn(self, *, allow_tools: bool) -> TurnResult:
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": self._system_prompt}]},
            "contents": self._contents,
            "tools": self._tools,
            "toolConfig": {
                "functionCallingConfig": {
                    "mode": "AUTO" if allow_tools else "NONE",
                }
            },
        }
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        resp = post_json(
            self._api_url,
            headers,
            payload,
            provider_label="gemini",
        )

        if resp.get("error"):
            raise RuntimeError(f"Gemini API error: {resp['error']}")

        usage_raw = resp.get("usageMetadata") if isinstance(resp.get("usageMetadata"), dict) else {}
        prompt_tokens = int(usage_raw.get("promptTokenCount") or 0)
        cache_hit = int(usage_raw.get("cachedContentTokenCount") or 0)
        candidates_tokens = int(usage_raw.get("candidatesTokenCount") or 0)
        thoughts_tokens = int(usage_raw.get("thoughtsTokenCount") or 0)
        # Match OpenAI/DeepSeek convention: completion = total billable output (candidates + thoughts)
        completion_tokens = candidates_tokens + thoughts_tokens
        total_tokens = int(usage_raw.get("totalTokenCount") or (prompt_tokens + completion_tokens))
        usage = TokenUsage(
            prompt=prompt_tokens,
            cache_hit=cache_hit,
            completion=completion_tokens,
            reasoning=thoughts_tokens,
            total=total_tokens,
        )

        candidates = resp.get("candidates")
        if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
            prompt_feedback = resp.get("promptFeedback") if isinstance(resp.get("promptFeedback"), dict) else {}
            block_reason = prompt_feedback.get("blockReason")
            raise RuntimeError(f"Gemini returned no candidates (blockReason={block_reason}).")

        cand = candidates[0]
        content = cand.get("content") if isinstance(cand.get("content"), dict) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []

        tool_calls: list[ToolCall] = []
        text_chunks: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if "functionCall" in part:
                fc = part["functionCall"]
                if isinstance(fc, dict):
                    tool_calls.append(
                        ToolCall(
                            id=str(fc.get("id") or ""),
                            name=str(fc.get("name") or ""),
                            args=fc.get("args") if isinstance(fc.get("args"), dict) else {},
                        )
                    )
            elif not part.get("thought") and isinstance(part.get("text"), str):
                text_chunks.append(part["text"])

        text = "".join(text_chunks).strip()
        finish_reason = cand.get("finishReason")
        if not text and not tool_calls:
            raise RuntimeError(f"Gemini returned empty content (finishReason={finish_reason}).")
        if finish_reason == "MAX_TOKENS" and not tool_calls and text:
            print("Warning: Gemini output truncated due to MAX_TOKENS.", file=sys.stderr)
            text += "\n\n> ⚠️ Output truncated by model MAX_TOKENS limit."

        return TurnResult(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            model=str(resp.get("modelVersion") or self._model),
            raw_assistant=content,
        )

    def append_tool_results(
        self,
        turn: TurnResult,
        results: list[tuple[ToolCall, str]],
    ) -> None:
        # Echo raw model content verbatim to preserve thoughtSignature on thinking models
        raw_content = turn.raw_assistant
        if isinstance(raw_content, dict) and raw_content.get("parts"):
            echoed = dict(raw_content)
            if not echoed.get("role"):
                echoed["role"] = "model"
            self._contents.append(echoed)

        response_parts = []
        for call, output in results:
            fr: dict[str, Any] = {
                "name": call.name,
                "response": {"result": output},
            }
            if call.id:
                fr["id"] = call.id
            response_parts.append({"functionResponse": fr})

        self._contents.append({"role": "user", "parts": response_parts})

    def append_final_nudge(self, nudge_text: str) -> None:
        if self._contents and self._contents[-1].get("role") == "user":
            parts = self._contents[-1].get("parts")
            if isinstance(parts, list):
                parts.append({"text": nudge_text})
            else:
                self._contents[-1]["parts"] = [{"text": nudge_text}]
        else:
            self._contents.append({"role": "user", "parts": [{"text": nudge_text}]})
