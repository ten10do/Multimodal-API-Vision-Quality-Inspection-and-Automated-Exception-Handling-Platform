"""LLM provider abstraction (9C).

`LlmProvider` is a minimal protocol over an OpenAI-compatible chat
completions endpoint with tool calling. Business logic never binds to a
specific vendor; `FakeLlmProvider` makes tests/E2E deterministic and
offline (no paid API), while `OpenAiLlmProvider` talks to any
OpenAI-compatible server configured via env vars. The API key is read from
the environment only and never enters git.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

logger = logging.getLogger(__name__)

# rough per-1k-token cost model for the estimated_cost field (9L). These are
# configurable estimates, not vendor billing.
INPUT_TOKEN_RATE = 0.0001   # USD per 1k input tokens
OUTPUT_TOKEN_RATE = 0.0003  # USD per 1k output tokens


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LlmResult:
    message: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class LlmProviderError(Exception):
    pass


class LlmProvider(Protocol):
    name: str

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> LlmResult: ...


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        input_tokens / 1000.0 * INPUT_TOKEN_RATE + output_tokens / 1000.0 * OUTPUT_TOKEN_RATE,
        6,
    )


class OpenAiLlmProvider:
    """OpenAI-compatible chat completions + tool calling over HTTP."""

    name = "openai"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout_seconds

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> LlmResult:
        import httpx

        started = time.perf_counter()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": 0.2,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
            body["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=body
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001 - any transport/HTTP failure
            raise LlmProviderError(f"llm request failed: {exc}") from exc

        latency = (time.perf_counter() - started) * 1000.0
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args))
        usage = data.get("usage") or {}
        return LlmResult(
            message=content,
            tool_calls=calls,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=round(latency, 2),
        )


class FakeLlmProvider:
    """Deterministic offline provider (tests + E2E + offline eval).

    Two modes:
      - `script`: an explicit list of steps. Each step is either
        {"tool_calls": [{"name": ..., "arguments": {...}}]} or
        {"message": "..."}. The service consumes them in order.
      - auto routing: picks tools from the last user message by keyword and
        then produces a grounded, evidence-templated final answer.

    The auto final answer is built ONLY from the tool results passed in
    `messages` (role="tool"), so it is deterministic and grounded by
    construction.
    """

    name = "fake"

    # Action-phrase write intent (9A). Analysis wording like "为什么被剔除"
    # must NOT be blocked; commanding wording like "请把…放行 RELEASE" must.
    WRITE_RE: list[str] = [
        r"请把.*放行", r"把.*放行", r"放行.*RELEASE", r"执行\s*release",
        r"promote\s*到", r"晋升到", r"回滚到", r"修改规则", r"resolve",
        r"写数据库", r"删除", r"放行这个",
    ]

    # ordered canonical routing mirroring the eval dataset's expected tools
    ROUTES: list[tuple[list[str], list[str]]] = [
        (["事件链", "工业事件"], ["get_industrial_events", "get_inspection_detail"]),
        (["被剔除", "追溯", "为什么产品", "产品 P", "这条检验", "insp-"], ["get_inspection_detail", "get_product_history"]),
        (["batch", "批次"], ["get_batch_quality"]),
        (["plc 故障", "NACK", "COMMAND_FAILED", "SAFE_HOLD"], ["get_plc_fault_summary"]),
        (["mes", "MES"], ["get_mes_sync_summary"]),
        (["drift", "漂移", "模型", "准确率", "延迟 p95", "错误率", "模型表现"], ["get_model_metrics", "get_drift_status", "get_review_metrics"]),
        (["review 率"], ["get_quality_summary", "get_review_backlog"]),
        (["复核", "积压", "backlog", "人工", "确认率", "纠正率"], ["get_review_backlog", "get_review_metrics"]),
        (["增长最快"], ["get_defect_distribution", "get_defect_trend"]),
        (["缺陷", "scratches", "crazing"], ["get_defect_distribution", "get_defect_trend"]),
        (["line", "产线", "工位", "station"], ["compare_production_lines", "get_quality_summary"]),
        (["趋势", "良率", "摘要", "概况", "summary", "今天"], ["get_quality_summary", "get_yield_trend"]),
    ]

    def __init__(self, script: list[dict] | None = None) -> None:
        self.script = script or []
        self._step = 0
        self.calls_seen: list[ToolCall] = []

    def reset(self, script: list[dict] | None = None) -> None:
        self.script = script or []
        self._step = 0
        self.calls_seen = []

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> LlmResult:
        if self._step < len(self.script):
            step = self.script[self._step]
            self._step += 1
            calls = [ToolCall(name=t["name"], arguments=t.get("arguments", {})) for t in step.get("tool_calls", [])]
            self.calls_seen.extend(calls)
            return LlmResult(
                message=step.get("message", ""),
                tool_calls=calls,
                input_tokens=len(system) // 4,
                output_tokens=len(step.get("message", "")) // 4,
                latency_ms=5.0,
            )
        # auto routing: decide tool calls from the last user message
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        low = last_user.lower()
        # safety: write-intent requests get NO tools (9A)
        import re

        if any(re.search(p, low) for p in self.WRITE_RE):
            return LlmResult(
                message="我是只读分析助手，无法执行放行/剔除/晋升/回滚等写操作。我可以分析原因并给出建议。",
                tool_calls=[], input_tokens=64, output_tokens=32, latency_ms=5.0,
            )
        wanted: list[str] = []
        for keywords, tools_to_call in self.ROUTES:
            if any(k.lower() in low for k in keywords):
                wanted = tools_to_call
                break
        if not wanted:
            wanted = ["get_quality_summary"]
        # only request tools that exist and that we have NOT yet seen a result for
        have_tool_results = {
            (m.get("tool_call_id") or "") for m in messages if m.get("role") == "tool"
        }
        if have_tool_results:
            # we already have results -> emit the grounded final answer
            return LlmResult(
                message=_fake_grounded_answer(messages),
                tool_calls=[],
                input_tokens=64,
                output_tokens=96,
                latency_ms=5.0,
            )
        calls = [ToolCall(name=n, arguments={}) for n in wanted]
        self.calls_seen.extend(calls)
        return LlmResult(message="", tool_calls=calls, input_tokens=64, output_tokens=16, latency_ms=5.0)


def _fake_grounded_answer(messages: list[dict]) -> str:
    """Template a final answer strictly from tool results already present.

    Only NUMERIC tool values are echoed (percentages/deltas come out as the
    raw fractions so the grounding validator can map them); string values
    like time_window dates are intentionally NOT echoed to avoid date
    numbers being treated as unsupported claims."""
    numbers: list[str] = []
    errors: list[str] = []
    for m in messages:
        if m.get("role") != "tool":
            continue
        payload = m.get("content", "{}")
        try:
            data = json.loads(payload) if isinstance(payload, str) else payload
        except Exception:  # noqa: BLE001
            data = {}
        if isinstance(data, dict) and data.get("error"):
            errors.append(f"工具 {data.get('tool')} 出错: {data.get('error')}")
            continue
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numbers.append(f"{v}")
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            for sv in item.values():
                                if isinstance(sv, (int, float)) and not isinstance(sv, bool):
                                    numbers.append(f"{sv}")
    summary = "已完成只读分析。关键指标："
    summary += ("、" .join(numbers[:10]) if numbers else "（无可用数值）")
    summary += "。所有数字均来自工具结果。"
    if errors:
        summary += " 注意：" + "；".join(errors[:3])
    return summary


def get_llm_provider() -> LlmProvider:
    """Build the configured provider (9C): 'fake' -> FakeLlmProvider,
    'openai' -> OpenAiLlmProvider from env settings."""
    from ..config import get_settings

    s = get_settings()
    if s.llm_provider == "openai":
        return OpenAiLlmProvider(
            base_url=s.llm_base_url,
            model=s.llm_model,
            api_key=s.llm_api_key,
            timeout_seconds=s.llm_timeout_seconds,
        )
    return FakeLlmProvider()


ProviderFactory: Callable[..., Awaitable[LlmProvider]] | None = None
