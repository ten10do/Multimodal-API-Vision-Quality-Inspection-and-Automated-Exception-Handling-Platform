# Demo Script (5–8 minutes)

Real system, live data (demo seed + camera simulator). Screenshots:
[screenshots/](screenshots/).

| Time | Segment | What to show / say |
|---|---|---|
| 0:00 | Project goal | "工业 AI 质检平台：从相机帧到现场执行、再到模型治理与只读质量分析助手，全链路可运行、可测试。" |
| 0:30 | Architecture | `docs/architecture.md` 图：四类职责边界（AI 决策 / 人工决策 / 最终质量结果 / 工业执行）。强调 DB 是唯一事实源。 |
| 1:00 | Live inspection | Overview 页：camera simulator 持续产生检验，WebSocket 实时刷新（`01-overview.png`）。 |
| 1:40 | YOLO + PatchCore | Inspection Detail：已知缺陷 bbox + 置信度（`03-traceability`/`04-inspection-detail-bbox`）；Review Detail 的 anomaly heatmap（`10-review-anomaly`）。说明融合 → UNKNOWN_ANOMALY。 |
| 2:20 | Human review | Review Queue：claim → 查看 AI 证据 → CONFIRM/CORRECT/PASS → 审计记录（`06/07/08/09`）。强调 AI 证据不被覆盖。 |
| 3:10 | PLC/MES | Industrial Status：REVIEW→HOLD→HELD；human PASS→RELEASE；FAIL→REJECTED；故障注入 offline→SAFE_HOLD 绝不 RELEASE（`11-phase7-industrial-detail`）。MES 同步状态。 |
| 4:00 | MLOps | Model Operations：当前 PRODUCTION、deployment version、metrics、drift（`12-phase8-modelops`）。演示 promote 被 domain gate 拒绝（MVTec PatchCore AUROC 1.0 不能晋升钢材模型）。 |
| 5:00 | Quality Copilot | Copilot 页：示例问题（今日良率 / 哪种缺陷最多 / 产品为何被剔除 / 模型是否漂移）→ 展示 Evidence / Time Window / Tools Used；演示 "请放行产品" → 只读拒绝。 |
| 6:00 | Failure handling | `docs/failure-recovery.md`：inference down / PLC timeout / MES 500 / checksum mismatch / rollback——每个都说 expected behavior + recovery。 |
| 7:00 | Engineering highlights | 8.8× E2E 优化；OPC UA namespace bug + fail-fast gate；NOT_INTEGRATED vs SAFE_HOLD；deployment/model/dataset 版本分离；Copilot 只读 + grounding（unsupported=0）。 |

## Runtime cues

- 先 `bash scripts/demo_up.sh`，等 `OVERALL: READY`。
- Copilot 演示用离线 provider（确定性）；说明真实 LLM gate 为
  `REAL_LLM_GATE_NOT_RUN`（外部端点待定，不伪造 PASS）。
- 故障演示可选：停 inference → `/ready` 失败；`?mode=offline` PLC → SAFE_HOLD。
