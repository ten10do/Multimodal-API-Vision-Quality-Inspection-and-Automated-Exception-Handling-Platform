# IndustrialVision-QC

面向智能制造的工业机器视觉质检与质量闭环平台（求职作品集项目）。

## 定位

完整的工业质检软件链路：模拟相机采集 → YOLO 已知缺陷检测 → PatchCore 未知异常检测 → OCR 产品编号 → 质量规则引擎 → PASS/REVIEW/FAIL → 人工复核 → PLC/MES 模拟联动 → 质量追溯 → 模型监控 → 数据反馈 → 持续优化。

全部工业设备以软件模拟，总成本控制在 300 至 500 元以内。

## 技术栈

PyTorch / YOLOv8 / PatchCore / PaddleOCR / FastAPI / SQLAlchemy / WebSocket / Redis / PostgreSQL / MinIO / React / TypeScript / ECharts / Docker Compose / pytest / GitHub Actions / MLflow / Prometheus / Grafana

## 当前状态

- [x] Phase 0 需求与架构基线（见 [docs/00-phase0-baseline.md](docs/00-phase0-baseline.md)）
- [x] Phase 1 Vision MVP（YOLOv8s + NEU-DET，见 [docs/02-phase1-report.md](docs/02-phase1-report.md)）
- [x] Phase 2 Backend MVP（FastAPI + PostgreSQL + Rule Engine + Inference HTTP API，见本文件与 docs）
- [x] Phase 3 Realtime Pipeline（Camera Simulator + Orchestrator + WebSocket，见 [docs/05-phase3-benchmark.md](docs/05-phase3-benchmark.md)）
- [ ] Phase 4 Dashboard

## 本地启动（Phase 2）

依赖：Python 3.11 venv（已冻结在 `model-training/requirements.txt` / `backend/requirements.txt`）、Docker Desktop。

```bash
# 1. 基础设施：PostgreSQL（Docker）
docker compose up -d postgres

# 2. 数据库迁移（从空库完整建表）
cd backend
../.venv/Scripts/python.exe -m alembic upgrade head
cd ..

# 3. 种子质量规则（幂等）
../.venv/Scripts/python.exe scripts/seed_quality_rules.py

# 4. 推理服务（独立 HTTP 进程，默认 8100 端口）
cd inference-service
../.venv/Scripts/python.exe -m uvicorn inference_app.api:app --host 0.0.0.0 --port 8100
cd ..

# 5. 后端 API（默认 8000 端口）
cd backend
../.venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
cd ..
```

验证：

- API 文档 http://localhost:8000/docs
- 推理服务健康检查 http://localhost:8100/health

上传一张测试图片触发完整链路：

```bash
curl -s -X POST http://localhost:8000/api/v1/inspections \
  -F "file=@model-training/datasets/neu-det-yolo/test/images/crazing_101.jpg" \
  -F "product_id=NEU-0001" -F "production_line=line-a" -F "station=qc-01"
```

## Realtime Pipeline 启动（Phase 3）

```bash
# 1. 推理服务（独立进程，GPU）
cd inference-service
../.venv/Scripts/python.exe -m uvicorn inference_app.api:app --port 8100
cd ..

# 2. 后端（指向容器 PostgreSQL）
cd backend
IVQC_DATABASE_URL=postgresql+asyncpg://vision_qc:vision_qc@127.0.0.1:5433/industrialvision_dev \
IVQC_INFERENCE_SERVICE_URL=http://127.0.0.1:8100 \
../.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
cd ..

# 3. Camera Simulator + Orchestrator（通过 Backend API 进入系统）
.venv/Scripts/python.exe -m simulator.run_pipeline \
  --images 60 --interval-ms 500 --workers 2 --queue-size 20 \
  --backend-url http://127.0.0.1:8000 --batch batch-demo-001
```

实时事件订阅（WebSocket）：

```bash
# 用 websocat 或任意 WS 客户端连接
ws://127.0.0.1:8000/api/v1/ws/inspections
```

实时指标：`GET http://127.0.0.1:8000/api/v1/realtime/status`。

测试库准备（可复现，仅操作容器内 industrialvision_test）：

```bash
.venv/Scripts/python.exe scripts/prepare_test_db.py --recreate
```

## 测试

```bash
# 单元层（默认，不加载模型、不依赖外部服务）
.venv/Scripts/python.exe -m pytest

# 集成层（需推理服务 + PostgreSQL 已启动）
.venv/Scripts/python.exe -m pytest -m integration

# GPU 层
.venv/Scripts/python.exe -m pytest -m gpu
```

## 目录

```
backend/            FastAPI 模块化单体
inference-service/  YOLO + PatchCore + OCR 推理服务
simulator/          Camera / PLC / MES 模拟器
frontend/           React + Vite + TypeScript + ECharts
model-training/     训练与评估脚本
monitoring/         监控配置
docs/               架构与工程文档
legacy/             已归档的 vision-qc-agent（多模态 LLM 路线）
```
