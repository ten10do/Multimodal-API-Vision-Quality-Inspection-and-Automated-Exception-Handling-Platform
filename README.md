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
- [x] Phase 4 Frontend Dashboard（React + Vite + TypeScript + ECharts，见 [docs/06-phase4-dashboard.md](docs/06-phase4-dashboard.md)）
- [ ] Phase 5 Dashboard Interactive

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

## Frontend Dashboard 启动与演示（Phase 4）

依赖：Node 22.22.2 + npm。依赖安装使用锁文件（CI 验证通过，详见 Gate 2 说明）。

```bash
cd frontend
# 标准安装（基于 package-lock.json，CI 干净 runner 验证 npm ci / npm test / npm run build 全绿）
npm ci --no-audit --no-fund
npm exec playwright install chromium

# 启动顺序：推理服务 → 后端 → 模拟器 → 前端
cd ../inference-service
../.venv/Scripts/python.exe -m uvicorn inference_app.api:app --port 8100

cd ../backend
IVQC_DATABASE_URL=postgresql+asyncpg://vision_qc:vision_qc@127.0.0.1:5433/industrialvision_test \
IVQC_INFERENCE_SERVICE_URL=http://127.0.0.1:8100 \
../.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000

cd ..
.venv/Scripts/python.exe -m simulator.run_pipeline --interval-ms 500 --workers 2 --backend-url http://127.0.0.1:8000 --batch demo-p4 --loop

cd frontend
"C:/Users/EDY/.workbuddy/binaries/node/versions/22.22.2/node.exe" "C:/Program Files/nodejs/node_modules/npm/bin/npm-cli.js" run dev
# 浏览器访问 http://127.0.0.1:5173
```

演示截图（含真实链路数据）：

```bash
cd frontend
NODE_OPTIONS="" "C:/Users/EDY/.workbuddy/binaries/node/versions/22.22.2/node.exe" e2e/demo.cjs
# 截图输出到 docs/screenshots/01-overview.png ... 05-overview-running.png
```

前端测试：

```bash
"C:/Users/EDY/.workbuddy/binaries/node/versions/22.22.2/node.exe" "C:/Program Files/nodejs/node_modules/npm/bin/npm-cli.js" test          # vitest 27 项
NODE_OPTIONS="" "C:/Users/EDY/.workbuddy/binaries/node/versions/22.22.2/node.exe" "C:/Program Files/nodejs/node_modules/npm/bin/npm-cli.js" exec playwright test  # 浏览器 E2E 5 项
```

## Frontend 依赖可复现性（Gate 2）

标准安装路径为 `npm ci`（基于已提交的 `frontend/package-lock.json`），已在
GitHub Actions 双 runner（ubuntu-latest + windows-latest）上验证：

```
npm ci → verify proxy-agents dist → npm test → npm run build  全部通过
```

本机（Windows + npm 11.16.0 + WorkBuddy safe-delete shim 注入 NODE_OPTIONS）
存在宿主特异性问题：`http-proxy-agent` / `agent-base` / `https-proxy-agent`
三个包的 `dist/index.js` 在 npm 解包时丢失，导致 vitest 无法启动。证据：

- 干净目录 + 全新 npm cache 下 `npm ci` 复现（非缓存问题）；
- 同一 lockfile 在 GitHub Actions 干净 runner（含 windows-latest）上
  `npm ci` 完整正常，dist 文件校验通过；
- 根因指向本机 host FS 层 / safe-delete shim 对 npm 解包的干扰。

本机复现该问题时，可先应用临时补丁（从 registry tarball 补齐 3 个文件），
该补丁**不是**标准安装步骤；标准步骤以 CI 验证的 `npm ci` 为准。此外本机
运行 vite build / playwright 等会自行清理临时目录的工具时，需
`NODE_OPTIONS=""` 以避开 safe-delete shim 对临时目录回收的拦截。

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
