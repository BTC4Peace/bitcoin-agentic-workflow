"""Static provider registry. Add new AI agents here with minimal boilerplate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reviewer.providers.base import BaseProvider
from reviewer.providers.gemini import GeminiProvider
from reviewer.providers.openai_compat import OpenAICompatProvider


@dataclass(frozen=True)
class ProviderSpec:
    kind: str  # "openai_compat" or "gemini"
    default_model: str
    api_url: str = ""


# Hardcoded allowlist of supported providers and HTTPS endpoints (no user-supplied URLs)
PROVIDERS: dict[str, ProviderSpec] = {
    "deepseek": ProviderSpec(
        kind="openai_compat",
        default_model="deepseek-flash",
        api_url="https://api.deepseek.com/chat/completions",
    ),
    "gemini": ProviderSpec(
        kind="gemini",
        default_model="gemini-4-neon",
    ),
}


def create_provider(
    provider_name: str,
    model: str,
    api_key: str,
    tool_specs: list[dict[str, Any]],
) -> BaseProvider:
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider_name!r}")
    spec = PROVIDERS[provider_name]
    chosen_model = model.strip() or spec.default_model

    if spec.kind == "openai_compat":
        return OpenAICompatProvider(
            provider_name=provider_name,
            api_url=spec.api_url,
            model=chosen_model,
            api_key=api_key,
            tool_specs=tool_specs,
        )
    if spec.kind == "gemini":
        return GeminiProvider(
            model=chosen_model,
            api_key=api_key,
            tool_specs=tool_specs,
        )
    raise ValueError(f"Unknown provider kind: {spec.kind!r}")
