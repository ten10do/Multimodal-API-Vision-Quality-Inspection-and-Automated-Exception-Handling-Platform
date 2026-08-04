# IndustrialVision-QC

面向智能制造的工业机器视觉质检与质量闭环平台（求职作品集项目）。

## 定位

完整的工业质检软件链路：模拟相机采集 → YOLO 已知缺陷检测 → PatchCore 未知异常检测 → OCR 产品编号 → 质量规则引擎 → PASS/REVIEW/FAIL → 人工复核 → PLC/MES 模拟联动 → 质量追溯 → 模型监控 → 数据反馈 → 持续优化。

全部工业设备以软件模拟，总成本控制在 300 至 500 元以内。

## 技术栈

PyTorch / YOLOv8 / PatchCore / PaddleOCR / FastAPI / SQLAlchemy / WebSocket / Redis / PostgreSQL / MinIO / React / TypeScript / ECharts / Docker Compose / pytest / GitHub Actions / MLflow / Prometheus / Grafana

## 当前状态

- [x] Phase 0 需求与架构基线（见 [docs/00-phase0-baseline.md](docs/00-phase0-baseline.md)）
- [ ] Phase 1 Vision MVP

## 快速开始（开发中，Phase 2 后可用）

```bash
docker compose up --build
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
