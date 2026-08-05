# Phase 5 Human-in-the-loop Review

人工复核闭环（5A-5M）。所有数据来自真实 Backend REST + WebSocket；AI
原始判断永久保留（5F），Human decision 与 Final Quality Result 独立
保存（5G），并发 claim 安全（5D），Training Candidate 可导出（5J）。

## DB Schema

新增三张表（迁移 0004）：

- `review_tasks`：每个 inspection 最多一个 active（PENDING/IN_REVIEW）任务；
  通过部分唯一索引（`status != 'RESOLVED'`）在 DB 层强制（5B）。字段包含
  priority / assigned_to / claimed_at / resolved_at / version（乐观锁）/ AI 快照。
- `review_decisions`：人工决策记录，创建后不可修改；如需修订走
  `review_corrections` 追加审计记录（5F）。
- `review_corrections`：审计修订（who / what / when / why）。

`inspections` 增加 `final_quality_result`（业务最终事实），`quality_result`
保留 AI 原始判断（创建时固化）。非 REVIEW 的 inspection 在创建时
`final = quality_result`；REVIEW 留空，待人工 resolve 后写入。

`ai_defects_snapshot` 在任务创建时冻结（JSON 数组），后续数据变化不影响
已归档的原始判断。

## Review Lifecycle

```
REVIEW inspection persisted
  -> ReviewTask PENDING (idempotent; system FAILED never enters)
       ReviewEvent review.created  (WS 通知; DB 仍是事实来源)

Reviewer Claim (REST POST /reviews/{id}/claim)
  -> ReviewTask IN_REVIEW (乐观锁 + 部分唯一索引；并发失败者 409 already_claimed)
       ReviewEvent review.claimed

Reviewer Resolve (REST POST /reviews/{id}/resolve)
  -> ReviewTask RESOLVED
  -> ReviewDecision 写入（ai_quality_result / ai_defects_snapshot / human_decision / human_label / final_quality_result / reason）
  -> Inspection.final_quality_result = final_quality_result
       ReviewEvent review.resolved
```

Resolve 映射：PASS → PASS；CONFIRM_DEFECT / CORRECT_DEFECT / OTHER_DEFECT → FAIL。
CONFIRM_DEFECT / CORRECT_DEFECT / OTHER_DEFECT 必须填 human_label（422）。
不属于当前 reviewer 的 resolve 返回 409 not_owner。已 RESOLVED 的二次 resolve
返回 409 already_resolved，原始 decision 永不覆盖。

Resolve 完成后如需修订：POST `/reviews/{id}/corrections` 追加 ReviewCorrection，
原 decision 不变（5F）。

## Concurrency Strategy

两个并发 claim 场景，DB 层面安全：

- 共享 DB（PostgreSQL / 文件 SQLite）：条件 UPDATE WHERE status='PENDING'；
  第一个事务 COMMIT 之后，第二个事务的 WHERE 重评估 → 0 rows → 409。
- 内存 SQLite：每个 connection 独立 DB，需要
  StaticPool + per-request sessions 才能真实测试；测试用 `tmp_path/` 文件 DB
  + `connect_args={"timeout": 30}` + 独立 session factory，模拟真实并发。

测试覆盖：两个 coroutine `asyncio.gather` 同时 POST claim，断言 200/409
各一个。

## AI vs Human vs Final Result

- `Inspection.quality_result`：AI 原始判断（创建时固化，永不修改）。
- `Inspection.final_quality_result`：业务最终事实。
  - 非 REVIEW：在创建时由 Rule Engine 写入（PASS → PASS，FAIL → FAIL）。
  - REVIEW：留空，等待人工 resolve。
- `ReviewDecision.ai_quality_result` + `ai_defects_snapshot`：任务创建时
  冻结的 AI 原始判断（与 inspection 的 quality_result 一致）。
- `ReviewDecision.human_decision` + `human_label` + `final_quality_result`：
  人工决策与最终质量结果。

完整追溯链：

```
inspection (model_version / rule_version / defects / ai quality_result)
  -> review_task (ai_defects_snapshot / priority)
    -> review_decision (ai snapshot + human_decision + final_quality_result)
    -> review_corrections (audit revisions, if any)
```

## Review Queue UI

新页面 `Review Queue`（导航第 3 个 tab）：

- 指标行：Pending / Resolved / Avg Wait / Review Rate / Agreement / Override /
  Corrected Labels。
- 队列表：product_id / image 缩略图 / AI defect / confidence / severity / rule /
  line / station / 等待时间 / assigned / status。
- 点击行打开详情 modal：原图 + SVG Bounding Box + AI Prediction（固化快照表）+
  Product Metadata + Review Controls（Claim / 4 个 Decision radio / human_label
  输入 / reason 输入 / Resolve 按钮）。
- 状态徽章：PENDING（待认领）/ IN_REVIEW（复核中）/ RESOLVED（已复核）。
- 冲突处理：409（already_claimed / already_resolved / not_owner）显示在错误条。

## Audit Trail

已 RESOLVED 决策的修订通过 `POST /reviews/{id}/corrections` 追加记录：

- who：reviewer（明确 user identifier，第一版不强制 RBAC）
- what：field_changed + old_value + new_value
- when：created_at（墙钟）
- why：reason

测试断言：追加 correction 后 `task.decision.human_decision` 保持 PASS，
`task.decision.corrections.length == 1`。

## Training Candidate

`GET /api/v1/training-candidates?kind=corrected|disagreed|low_confidence&format=json|csv`

- corrected：human_decision in (CORRECT_DEFECT, OTHER_DEFECT)
- disagreed：human_decision in (PASS, CORRECT_DEFECT, OTHER_DEFECT)
- low_confidence：AI top confidence < 0.6

字段：inspection_id, image_url (StorageService), ai_label, human_label,
ai_confidence, agreement, review_reason, model_version, timestamp。

不自动触发训练（5J），仅导出 manifest。

## Review Metrics（5K）

`GET /api/v1/reviews-metrics` 显式语义：

| 字段 | 语义 |
|---|---|
| `pending_review_count` | PENDING + IN_REVIEW |
| `pending` | PENDING |
| `in_review` | IN_REVIEW |
| `resolved` | RESOLVED |
| `average_review_wait_time_s` | mean(resolved_at - created_at) over resolved |
| `review_rate` | total tasks / completed inspections |
| `ai_human_agreement_rate` | CONFIRM_DEFECT / resolved |
| `override_rate` | 1 - agreement_rate |
| `corrected_label_count` | count(CORRECT_DEFECT) |
| `pass_overrides` | count(PASS) |

演示数据真实闭环产出：`pending 534 / resolved 13 / review_rate 0.4532 /
ai_human_agreement_rate 0.3846 / override_rate 0.6154 / corrected 5 /
pass_overrides 3 / avg_wait 176s`。

## WebSocket Events

`ReviewEvent` 复用了现有 `/api/v1/ws/inspections` 通道（5I）：

- `review.created` / `review.claimed` / `review.resolved`
- DB 是事实来源；WS 是通知；断线/重连通过 REST reconciliation 重建。
- 前端 `useInspectionSocket` 解析两类事件（inspection.* / review.*），
  `parseWsEvent` 联合类型守卫；Live 页过滤 inspection 事件，Review Queue 页
  接收 review 事件并触发 refetch。

## Frontend 测试（5L）

- vitest 33 通过（含 transforms 的 review 事件解析、决策校验、最终映射、
  waiting time；reviewQueue.test.tsx 队列渲染、Claim、Resolve、决策校验、
  409 冲突）。

## Playwright E2E（5M，6 通过）

- PASS override → DB 保留 AI REVIEW，final_quality_result = PASS
- CONFIRM_DEFECT → final FAIL
- CORRECT_DEFECT → final FAIL + training-candidates?kind=corrected 含该样本
- 并发 claim → 409 already_claimed 状态显示
- 全链路审计：每个 RESOLVED inspection 的 quality_result 保持 REVIEW，
  final_quality_result 等于人工 decision

## 已知问题

1. 并发测试在内存 SQLite 不可信；测试使用文件 SQLite + per-request sessions
   模拟真实并发。本机验证为 PostgreSQL 容器（industrialvision_test）。
2. Dashboard captured_total 偶发短暂 < completed（telemetry 2s 间隔），与
   Phase 4 结论一致；本阶段不修复。

Phase 5 完成后停止，不进入 Phase 6。