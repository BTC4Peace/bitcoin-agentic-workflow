"""OpenAI-compatible Chat Completions provider (DeepSeek, OpenAI, Mistral, Groq, etc.)."""

from __future__ import annotations

import sys
from typing import Any

from reviewer.http_client import post_json
from reviewer.providers.base import BaseProvider, TokenUsage, ToolCall, TurnResult


def _to_openai_tools(tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools = []
    for spec in tool_specs:
        params: dict[str, Any] = {
            "type": "object",
            "properties": spec["properties"],
        }
        if spec.get("required"):
            params["required"] = spec["required"]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": params,
                },
            }
        )
    return tools


class OpenAICompatProvider(BaseProvider):
    def __init__(
        self,
        *,
        provider_name: str,
        api_url: str,
        model: str,
        api_key: str,
        tool_specs: list[dict[str, Any]],
    ) -> None:
        self._provider_name = provider_name
        self._api_url = api_url
        self._model = model
        self._api_key = api_key
        self._tools = _to_openai_tools(tool_specs)
        self._messages: list[dict[str, Any]] = []

    def init_conversation(self, system_prompt: str, user_content: str) -> None:
        self._messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def call_turn(self, *, allow_tools: bool) -> TurnResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            "tools": self._tools,
            "tool_choice": "auto" if allow_tools else "none",
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        resp = post_json(
            self._api_url,
            headers,
            payload,
            provider_label=self._provider_name,
        )

        if resp.get("error"):
            raise RuntimeError(f"{self._provider_name} API error: {resp['error']}")

        usage_raw = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
        prompt_details = (
            usage_raw.get("prompt_tokens_details")
            if isinstance(usage_raw.get("prompt_tokens_details"), dict)
            else {}
        )
        completion_details = (
            usage_raw.get("completion_tokens_details")
            if isinstance(usage_raw.get("completion_tokens_details"), dict)
            else {}
        )
        cache_hit = int(
            usage_raw.get("prompt_cache_hit_tokens")
            or prompt_details.get("cached_tokens")
            or 0
        )
        usage = TokenUsage(
            prompt=int(usage_raw.get("prompt_tokens") or 0),
            cache_hit=cache_hit,
            completion=int(usage_raw.get("completion_tokens") or 0),
            reasoning=int(completion_details.get("reasoning_tokens") or 0),
            total=int(usage_raw.get("total_tokens") or 0),
        )

        choices = resp.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError(f"{self._provider_name} returned no choices.")

        finish_reason = choices[0].get("finish_reason")
        msg = choices[0].get("message")
        if not isinstance(msg, dict):
            raise RuntimeError(f"{self._provider_name} returned malformed message object.")

        raw_tcs = msg.get("tool_calls") or []
        parsed_tcs: list[ToolCall] = []
        if isinstance(raw_tcs, list):
            for tc in raw_tcs:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                parsed_tcs.append(
                    ToolCall(
                        id=str(tc.get("id") or ""),
                        name=str(fn.get("name") or ""),
                        args=fn.get("arguments", "{}"),
                    )
                )

        content_raw = msg.get("content")
        text = content_raw.strip() if isinstance(content_raw, str) else ""
        if not text and not parsed_tcs:
            raise RuntimeError(
                f"{self._provider_name} returned empty content (finish_reason={finish_reason}, refusal={msg.get('refusal')})."
            )
        if finish_reason == "length" and not parsed_tcs and text:
            print(f"Warning: {self._provider_name} output truncated (finish_reason=length).", file=sys.stderr)
            text += "\n\n> ⚠️ Output truncated by model max-token limit."

        return TurnResult(
            text=text,
            tool_calls=parsed_tcs,
            usage=usage,
            model=str(resp.get("model") or self._model),
            raw_assistant=msg,
        )

    def append_tool_results(
        self,
        turn: TurnResult,
        results: list[tuple[ToolCall, str]],
    ) -> None:
        msg = turn.raw_assistant or {}
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
        }
        if "reasoning_content" in msg and msg["reasoning_content"] is not None:
            assistant_msg["reasoning_content"] = msg["reasoning_content"]
        self._messages.append(assistant_msg)

        for call, output in results:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": output,
                }
            )

    def append_final_nudge(self, nudge_text: str) -> None:
        self._messages.append({"role": "user", "content": nudge_text})
