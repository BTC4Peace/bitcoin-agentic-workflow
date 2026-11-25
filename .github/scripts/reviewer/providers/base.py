"""Provider adapter protocol and normalized turn/usage structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    args: Any


@dataclass
class TokenUsage:
    prompt: int = 0
    cache_hit: int = 0
    completion: int = 0
    reasoning: int = 0
    total: int = 0

    def add(self, other: TokenUsage) -> None:
        self.prompt += other.prompt
        self.cache_hit += other.cache_hit
        self.completion += other.completion
        self.reasoning += other.reasoning
        self.total += other.total


@dataclass
class TurnResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    raw_assistant: Any = None


class BaseProvider(ABC):
    """Minimal interface any new AI provider adapter implements."""

    @abstractmethod
    def init_conversation(self, system_prompt: str, user_content: str) -> None:
        ...

    @abstractmethod
    def call_turn(self, *, allow_tools: bool) -> TurnResult:
        ...

    @abstractmethod
    def append_tool_results(
        self,
        turn: TurnResult,
        results: list[tuple[ToolCall, str]],
    ) -> None:
        ...

    @abstractmethod
    def append_final_nudge(self, nudge_text: str) -> None:
        ...
