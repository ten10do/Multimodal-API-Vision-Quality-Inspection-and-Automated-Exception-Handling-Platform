import { useState } from "react";
import { api } from "../../api/client";
import { MetricCard } from "../../components/MetricCard";
import { EmptyState } from "../../components/StateViews";
import type { CopilotResponse } from "../../types";

const EXAMPLES = [
  "今天整体良率如何？",
  "哪种缺陷增长最快？",
  "Line A 为什么异常？",
  "哪个工位需要优先检查？",
  "为什么产品 P-100 被剔除？",
  "当前模型是否存在漂移？",
  "当前人工复核积压多少？",
];

interface Entry {
  role: "user" | "assistant";
  content: string;
  result?: CopilotResponse;
  error?: string;
}

export function CopilotPage() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [entries, setEntries] = useState<Entry[]>([]);
  const [busy, setBusy] = useState(false);

  async function send(message: string) {
    const text = message.trim();
    if (!text || busy) return;
    setBusy(true);
    setEntries((prev) => [...prev, { role: "user", content: text }]);
    try {
      const res = await api.copilotQuery({ conversation_id: conversationId ?? undefined, message: text });
      setConversationId(res.conversation_id);
      setEntries((prev) => [...prev, { role: "assistant", content: res.message, result: res }]);
    } catch (err) {
      setEntries((prev) => [...prev, { role: "assistant", content: "", error: String(err) }]);
    } finally {
      setBusy(false);
      setInput("");
    }
  }

  const last = [...entries].reverse().find((e) => e.result)?.result;

  return (
    <div className="copilot-layout">
      <div className="panel copilot-chat">
        <div className="panel-head">
          <h3>Quality Copilot（只读分析助手）</h3>
          <span className="badge badge-ok">READ-ONLY</span>
        </div>
        <div className="copilot-examples">
          {EXAMPLES.map((q) => (
            <button key={q} className="chip-btn" disabled={busy} onClick={() => void send(q)}>
              {q}
            </button>
          ))}
        </div>
        <div className="copilot-thread">
          {entries.length === 0 && (
            <div className="empty-hint">选择上方示例问题，或直接输入自然语言问题。Copilot 只读分析，不会执行任何写操作。</div>
          )}
          {entries.map((e, i) => (
            <div key={i} className={`copilot-msg ${e.role}`}>
              <div className="copilot-msg-role">{e.role === "user" ? "你" : "Copilot"}</div>
              <div className="copilot-msg-body">{e.error ? `请求失败：${e.error}` : e.content}</div>
            </div>
          ))}
          {busy && <div className="copilot-msg assistant"><div className="copilot-msg-body">分析中…</div></div>}
        </div>
        <div className="copilot-input">
          <input
            value={input}
            placeholder="例如：为什么今天 Line A 良率下降？"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void send(input);
            }}
          />
          <button className="btn" disabled={busy || !input.trim()} onClick={() => void send(input)}>
            分析
          </button>
        </div>
      </div>

      <div className="copilot-evidence">
        {!last ? (
          <div className="panel"><EmptyState message="提问后，这里显示证据、指标、时间窗口与所用工具" /></div>
        ) : (
          <>
            <div className="panel">
              <h3>Evidence（全部来自工具结果）</h3>
              {last.evidence.length === 0 ? (
                <EmptyState message="无工具调用（纯回答）" />
              ) : (
                last.evidence.map((ev, i) => (
                  <div key={i} className="state-block" style={{ marginBottom: 6 }}>
                    <div className="mono">{ev.tool} · {ev.latency_ms}ms</div>
                    {ev.time_window && <div className="muted">window: {String(ev.time_window)}</div>}
                    <div className="mono small">{JSON.stringify(ev)}</div>
                  </div>
                ))
              )}
            </div>
            <div className="panel">
              <h3>Metrics / Time Window</h3>
              <div className="metric-grid">
                <MetricCard label="Confidence" value={last.confidence} />
                <MetricCard label="Tool Calls" value={last.latency.tool_call_count} />
                <MetricCard label="LLM Latency" value={`${Math.round(last.latency.llm_latency_ms)} ms`} />
                <MetricCard label="Total Latency" value={`${Math.round(last.latency.total_latency_ms)} ms`} />
                <MetricCard label="Input Tokens" value={last.latency.input_tokens} />
                <MetricCard label="Output Tokens" value={last.latency.output_tokens} />
                <MetricCard label="Est. Cost" value={`$${last.latency.estimated_cost_usd.toFixed(6)}`} />
              </div>
              {last.evidence.map((ev, i) => ev.time_window ? (
                <div key={i} className="state-block">window: {String(ev.time_window)}</div>
              ) : null)}
            </div>
            <div className="panel">
              <h3>Tools Used</h3>
              <div className="tool-chips">
                {last.tools_used.length === 0 ? (
                  <span className="muted">（无）</span>
                ) : (
                  last.tools_used.map((t) => (
                    <span key={t} className="chip">{t}</span>
                  ))
                )}
              </div>
              {last.limitations.length > 0 && (
                <>
                  <h3>Limitations</h3>
                  <ul className="limitations">
                    {last.limitations.map((l, i) => (
                      <li key={i} className="muted">{l}</li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
