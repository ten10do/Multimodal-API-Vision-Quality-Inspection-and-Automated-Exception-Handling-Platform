# Phase 2 汇报：Backend MVP

> IndustrialVision-QC | 日期 2026-08-04 | 本阶段结束，不进入 Phase 3。

## 0. 阶段目标

打通 Image → Inference Service(HTTP) → Standard Vision Contract → Backend → Quality Rule Engine → PostgreSQL → Inspection Result。不含 WebSocket / Dashboard / PLC / MES / PatchCore / MLOps / Copilot。

## 1. Docker 环境

| 项 | 状态 |
|---|---|
| Docker Desktop | 已安装（CLI 29.6.2，Compose v5.3.1），官方安装器 625MB 下载并提权安装 |
| Docker 引擎 | **阻塞项**：本机未安装 WSL2，Linux 引擎无法启动；启用 WSL 特性需管理员 + 系统重启（会中断会话，未擅自执行） |
| 用户操作 | 管理员运行 `wsl --install` → 重启 → 启动 Docker Desktop → `docker compose up -d postgres`（compose 文件已就绪） |
| PostgreSQL 验证替代 | 本机原生 PostgreSQL 16.4 实例（与 compose 镜像同大版本）承载全部 DB/E2E 验证 |

## 2. 数据库 schema 与迁移

5 张核心表（batches / products / inspections / defects / quality_rules），SQLAlchemy 2.0 async ORM + Alembic。

- ID 策略：UUID 主键 + 业务字符串 ID（product_id / inspection_id / batch_id）+ 唯一索引
- 时间戳：created_at / updated_at（server_default + onupdate）
- 枚举：status / quality_result / severity / batch status（native_enum=False，SQLite 测试兼容）
- 迁移 0001：从空库完整建表（已对全新 test 库验证）
- 迁移 0002：quality_rules 业务唯一约束 (defect_type, priority, rule_version)，防止重复规则；已实测 DB 拒绝重复插入
- 未用 create_all() 作为正式迁移方案（仅测试夹具使用）

## 3. 数据库隔离（本次修正）

| 库 | 用途 | 状态 |
|---|---|---|
| industrialvision_dev | 开发库 | alembic head(0002) + 8 条种子规则；冒烟数据 1 条 |
| industrialvision_test | 测试库 | 每次验证从空库重建：dropdb → createdb → alembic upgrade → seed → 测试 → 清理 |
| vision_qc | 早期临时库 | 已弃用，未删除（保留现场） |

已验证：测试运行后 dev 库 products=0 / inspections=0（零污染），仅 8 条种子规则。种子脚本幂等（ON CONFLICT DO NOTHING）：首跑 8 条、二跑 0 条。

## 4. Quality Rule Engine

独立模块 `backend/app/quality/engine.py`，输入仅来自 Vision Contract 的客观事实（class_name / confidence / defect_area_ratio），输出 PASS/REVIEW/FAIL + severity + matched_rule + rule_version + reason。

- 规则来自 DB（quality_rules 表），阈值不散落在业务代码
- 优先级语义明确：单缺陷取 priority 最小者匹配；多缺陷取最坏 action（FAIL>REVIEW>PASS）与最高 severity；无匹配规则 → REVIEW；无 detections → PASS（仅业务行为，见负样本策略）
- 覆盖用例：no detections→PASS、critical→FAIL、low confidence→REVIEW、area 阈值、多缺陷、规则优先级、禁用规则、版本化规则、通配规则

## 5. Inference Service HTTP API（Phase 2D）

独立 FastAPI 进程（`inference_app` 包），提供 GET /health、GET /ready、POST /v1/infer。实测：

```
/v1/infer → 200, cuda:0, yolov8s/phase1-baseline, 200x200, 28ms, detections 符合标准契约
```

- 后端仅通过 HTTP 调用（httpx），不 import 模型代码
- timeout（30s）/ 连接错误 / 非 200 / 契约校验失败 均有明确异常类型与映射
- request_id（X-Request-ID）贯穿，结构化日志

## 6. Backend API（Phase 2E/2F）

| 端点 | 说明 |
|---|---|
| POST /api/v1/inspections | multipart 图片 → 建产品/质检上下文 → HTTP 推理 → 契约校验 → 规则引擎 → 持久化；支持 idempotency_key（重放返回 200 同一条） |
| GET /api/v1/inspections/{id} | 质检详情（含缺陷与产品） |
| GET /api/v1/products/{id} | 产品 |
| GET /api/v1/products/{id}/inspections | 产品质检历史（可追溯） |
| GET/POST /api/v1/quality-rules, PATCH /{id} | 规则配置 |
| GET /health, /ready | 存活与就绪（ready 检查 DB） |

错误处理：统一 `{"error":{code,message,request_id}}`；inference timeout/connection→504、upstream 500/契约无效→502、非法图片→422、DB 失败→500（回滚）、重复请求→409/幂等 200。HTTP 冒烟实测：REVIEW/medium（crazing 0.42）。

## 7. 测试（分层报告）

| 套件 | 命令 | 结果 |
|---|---|---|
| 单元测试 | `pytest`（默认，收集阶段零 torch） | **40 passed**，3 连跑零崩溃 |
| 后端集成（真实推理） | `pytest -m integration`（e2e） | **1 passed** |
| Inference mock E2E | `pytest -m integration`（e2e_mock，桩推理+真实 PG，无 GPU） | **1 passed** |
| 推理服务集成 | `pytest -m integration`（predictor，真实权重） | **2 passed** |
| 完整 CPU 套件 | `pytest -m "not gpu"`（服务在线） | **44 passed** |
| 独立 GPU E2E | `cuda_smoke.py`（独立进程）+ `pytest -m gpu` | **PASSED / 1 passed** |

GPU 异常不影响 CPU/Backend 套件（gpu 层独立标记、独立命令）。pytest/torch 访问冲突记录与缓解见 docs/03；默认套件不依赖 CUDA。

## 8. 新增/修改文件（Phase 2 累计）

- packages/vision-contract/（共享契约包，backend 与 inference 共用）
- backend/app/（config/database/models/enums/schemas/quality/engine、inference/client、services、api、main）+ alembic（0001/0002）
- backend/tests/（quality_engine 10、inspection_api 10、e2e、e2e_mock、conftest）
- inference-service/inference_app/（api.py 包装 YoloPredictor）+ tests 分层
- docker-compose.yml（postgres）、.env.example、scripts/seed_quality_rules.py、scripts/smoke_infer.py
- docs/03-windows-torch-issue.md（新）、docs/02 已知问题修订、docs/01 数据策略修订
- pytest.ini（分层 marker + asyncio_mode）

## 9. Git 状态

- 新增提交：0372396（feat backend MVP）、70da3aa（fix 隔离/幂等/约束/包改名）
- `git diff --check` clean；工作树 clean；累计领先 origin/main 9 个提交
- 数据集与权重不在 git

## 10. 已知问题

1. **Docker 引擎未启动**：Docker Desktop 已装，但主机缺 WSL2，需管理员安装 + 重启（唯一阻塞项，详见第 1 节）
2. WSL 输出为 UTF-16 编码，诊断输出乱码（不影响结论）
3. 早期临时库 vision_qc 仍保留（含冒烟数据），后续可手动删除
4. pytest/torch Windows access violation：根因未定，靠收集阶段零 torch 规避（docs/03）

## 11. Phase 3 建议

- 安装 WSL2 并启动 Docker 引擎后，用 `docker compose up -d postgres` 验证容器版 PostgreSQL，替换原生实例
- Phase 3 实现 Camera Simulator（定时推流）+ WebSocket 实时推送 + 前端 Dashboard，直接复用本阶段的 API 与规则引擎
- 负样本与 PASS 语义已按约束落实，后续按 docs/01 策略积累同领域正常钢材样本
