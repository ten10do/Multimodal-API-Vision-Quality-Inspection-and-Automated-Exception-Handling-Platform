# Phase 4 Dashboard

Realtime 工业质检可视化。所有数字来自真实的 Backend REST API 与 WebSocket，
不存在静态假数据。

## 架构

```
frontend/  (React + Vite + TypeScript + ECharts)
  src/api/client.ts                 REST client（所有 fetch URL 集中）
  src/utils/transforms.ts           统一统计 transform（仪表盘数值唯一来源）
  src/utils/bbox.ts                 Bounding Box 坐标转换
  src/ws/socket.ts                  WebSocket client + 指数退避重连
  src/hooks/useInspectionSocket.ts   实时事件订阅（去重 + 边界 + 重连后 reconciliation）
  src/hooks/queries.ts              @tanstack/react-query REST 数据
  src/components/                   MetricCard / StatusBadge / Chart / BBoxImage / StateViews
  src/features/overview/            OverviewPage
  src/features/live/                LivePage
  src/features/trace/               TracePage（4E 追溯查询）
  src/features/inspection/          InspectionDetailPanel（图像 + BBox + 缺陷 + 完整质检信息）
```

Vite dev server 代理 /api/* 到 backend（含 WebSocket：`ws: true`）。浏览器只走
同源路径，避免 CORS。Dashboard 编译产物已通过 `tsc -b && vite build`
strict TS 检查（`noUnusedLocals` / `noUnusedParameters` / 等）。

## Realtime Metrics 语义（前置修正）

统一为：

| 字段 | 含义 |
|---|---|
| `captured_total` | 已产出捕获（simulator 计数） |
| `queued_current` | 当前在 bounded queue 中 |
| `processing_current` | 当前被 worker 处理中 |
| `completed_total` | 已完成质检 |
| `failed_total` | 系统处理失败（inference 不可达、retries 用尽） |
| `pass_total` / `review_total` / `fail_total` | 产品质量判定（完成后） |
| `queue_peak_depth` | 运行期峰值 |
| `current_throughput` | 60 秒窗口完成率 |

守恒定律（unit-tested）：

```
running:  captured_total == queued_current + processing_current + completed_total + failed_total
drained:  captured_total == completed_total + failed_total
quality:  pass_total + review_total + fail_total == completed_total
system FAILED  与  产品 FAIL  严格分离，绝不混用
```

## 视图

### Production Overview（首页）

`GET /api/v1/realtime/status` + `GET /api/v1/inspections?limit=300`。
指标卡：Total Inspected / Completed / System Failed / PASS / REVIEW / FAIL /
Yield Rate（PASS/COMPLETED，不含系统失败）/ Throughput / Queue Depth /
Processing Count / Avg E2E Latency / P95 E2E Latency / Model Version。
图表：Quality Result Distribution / Defect Type Distribution / Quality Trend /
Throughput-Latency Trend。数据为空时显示 Empty State，禁止随机填图。

### Live Inspection（实时事件）

WS `/api/v1/ws/inspections`（去重 + 上限 100）。连接状态以圆点 +
文字呈现：`connecting` / `connected` / `disconnected` / `reconnecting` /
`reconnected`，断线后通过 REST reconciliation 重新对齐事实。
产品 FAIL 与 SYSTEM FAILED 在视觉上明确分离：颜色 + 边框样式 + 标签文案
三重区分。同一页底部有 Backend REST 视图（事实来源）作为对照。

### Quality Traceability（追溯查询）

`GET /api/v1/inspections` 支持 product_id / inspection_id / batch_id /
quality_result / status / defect_type / production_line / station /
date_from / date_to / limit / offset。点击行打开 InspectionDetailPanel：
原图 + BBox SVG overlay（client-side，从 Vision Contract 算出，不依赖
服务器预渲染）+ Defects 列表 + 完整质检元信息（product / batch / line /
station / process_status / quality_result / severity / model_version /
rule_version / inference_latency_ms / created_at / error_message）。

## WebSocket 健壮性（4F）

- 有限退避重连（base 500ms，2^n 增长，封顶 10s）。
- 断线 → `reconnecting` / `disconnected`；恢复 → `reconnected` 并触发
  reconciliation（重新拉取 REST `/inspections?limit=50` 重建页面事实）。
- 持久化优先：WS 广播失败绝不回滚已落库的 inspection。

## Frontend 测试

```bash
cd frontend
npm test                # vitest 25 项（API transform、metric invariant、WS 解析、
                        #   重连逻辑、去重、边界列表、状态渲染、BBox 转换、空/错态）
npm run build           # tsc strict + vite build
NODE_OPTIONS="" npm exec playwright test   # 浏览器 E2E 5 项
```

浏览器 E2E（Playwright + chromium）覆盖真实链路：dashboard 在运行
simulator 时指标移动、Live 页 WS 收到事件、追溯查询打开带图与 BBox 的
详情、SYSTEM FAILED 与 FAIL 视觉分离。

## Phase 4 演示截图

`docs/screenshots/`：

| 文件 | 内容 |
|---|---|
| 01-overview.png | Production Overview 真实指标 + 4 个图表 |
| 02-live-inspection.png | 实时连接 connected + 最近事件 + Backend 视图 |
| 03-traceability.png | 追溯查询结果（PASS/REVIEW/FAIL 徽章） |
| 04-inspection-detail-bbox.png | 原图 + SVG Bounding Box + 缺陷表 + 完整质检元信息 |
| 05-overview-running.png | 演示进行中的 Overview |

## 已知问题

1. Vite 代理 WebSocket 需 `ws: true`（已在 vite.config.ts 中配置）；在生产构建
   用 Nginx 等反向代理时也需开启 Upgrade 转发。
2. WS 广播为异步 fire-and-forget，断线期间事件可能短暂缺失，恢复后通过
   REST reconciliation 重建事实。
3. Dashboard 的 `captured_total` 来自 telemetry 周期（每 2s），偶发短暂
   `completed > captured`（不到 2s 的滞后）；守恒定律在每个 telemetry
   tick 上严格成立。
4. Phase 5（交互式 Dashboard）前可收敛原生 PostgreSQL 至容器实例。
## Gate 1 修订（指标快照语义）

质量快照与运行期遥测彻底分离，API 返回 `snapshot_at` / `telemetry_at`：

- Quality / persisted facts（DB 单一事实）：`completed_total`、`failed_total`、
  `pass_total`、`review_total`、`fail_total`、`total_inspected`、`yield_rate`。
  不变量：`pass+review+fail == completed`、`total_inspected == completed + failed`。
- Runtime telemetry（管线视图，独立刷新）：`captured_total`、`queued_current`、
  `processing_current`、`queue_depth`、`throughput`。
- `captured_total` 不与 DB 计数器做跨时间戳守恒比较；Dashboard 单独以
  "Captured (Pipeline)" 卡片呈现，freshness bar 展示两类时间戳。
- 回归测试覆盖 2840 vs 2843 类矛盾（telemetry 滞后时质量快照保持自洽）。

## Gate 2 修订（依赖可复现性）

标准安装 = `npm ci`。GitHub Actions 双 runner（ubuntu + windows-latest）验证
`npm ci / npm test / npm run build` 全绿（workflow: .github/workflows/frontend-ci.yml）。
本机宿主特异性问题（npm 解包丢失 proxy-agents dist/index.js）已记录证据，
不作为标准安装步骤。
