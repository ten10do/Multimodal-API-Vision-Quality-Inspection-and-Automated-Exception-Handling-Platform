# IndustrialVision-QC（中文概览）

> 端到端工业 AI 质检平台：已知缺陷检测（YOLO）+ 未知异常检测（PatchCore）+
> 人工复核 + PLC/MES 工业集成 + 模型治理（MLOps）+ 证据 grounded 的只读质量分析
> Copilot。完整英文文档见 [README.md](README.md)。

## 能力矩阵（诚实声明）

| 能力 | 状态 |
|---|---|
| YOLO 钢材表面缺陷（NEU-DET） | ✅ 已验证（mAP50 0.82） |
| PatchCore MVTec-bottle benchmark | ✅ 已验证（Image AUROC 1.000） |
| **PatchCore 钢材域精度** | ⚠️ **未验证**（跨域 baseline，`steel_domain_validated=false`，禁止晋升） |
| 人工复核闭环 | ✅ 已验证 |
| HTTP PLC / OPC UA / MES 集成 | ✅ 已验证（幂等 + fail-safe，OPC UA 真实 E2E gate） |
| Model Registry / 单 PRODUCTION / Rollback | ✅ 已验证 |
| Deployment manifest + SHA-256 安全加载 | ✅ 已验证 |
| Drift 检测（PSI/KS，drift ≠ 性能下降） | ✅ 已验证 |
| Copilot 确定性 eval（46 例） | ✅ 已验证（unsupported claims = 0） |
| Copilot 真实 LLM smoke | ⏳ **待外部端点**（`REAL_LLM_GATE_NOT_RUN`，不伪造 PASS） |
| Redis / MinIO / Prometheus / Grafana / OCR | ❌ 未使用（by design） |

## 快速开始

```bash
docker compose up -d postgres        # 基础设施（Docker，PG 在 5433）
bash scripts/demo_up.sh              # 一键演示（模拟器+推理+后端+前端+seed）
bash scripts/run_clean.sh python scripts/health_check.py   # 健康检查
```

打开 http://127.0.0.1:5173。GPU 推理保留在宿主机（Windows + RTX），Docker 只跑
PostgreSQL（基础设施容器化、推理宿主化，见 [docs/10-phase7-report.md](docs/10-phase7-report.md)）。

## 关键决策速览

详见 [docs/engineering-decisions.md](docs/engineering-decisions.md)：
AI 结果 / 人工结果 / 最终结果三分离；DB 为唯一事实源、WebSocket 仅通知；
httpx 客户端复用带来 ~8.8× E2E 优化（561.7→63.6ms）；`NOT_INTEGRATED` ≠ `SAFE_HOLD`
（不伪造现场状态）；OPC UA namespace 硬编码 bug 由真实集成 gate 捕获；
静默 skip → fail-fast；deployment/model/dataset 版本分离；Copilot 证据优先且只读。

## 限制（不隐藏）

1. PatchCore 为跨域 MVTec baseline，钢材域精度未验证。
2. 真实 LLM Copilot smoke 待外部端点（REAL_LLM_GATE_NOT_RUN）。
3. PLC / MES / OPC UA 为软件模拟，现场部署需真实网关。
4. 对话上下文与 Copilot 统计缓存为内存实现（单 worker，TTL），未用 Redis。
5. torch 进程需经 `scripts/run_clean.sh`（Bash 会话 DLL 隔离）。

## 文档导航

- 阶段报告：[docs/](docs/)（Phase 0-12，含决策/benchmark/known-issues）
- 架构图：[docs/architecture.md](docs/architecture.md)
- Benchmark 汇总：[docs/benchmark-summary.md](docs/benchmark-summary.md)
- 测试矩阵：[docs/test-matrix.md](docs/test-matrix.md)
- 面试准备：[docs/interview-guide.md](docs/interview-guide.md) · [docs/resume-material.md](docs/resume-material.md)
- 演示脚本：[docs/demo-script.md](docs/demo-script.md)
