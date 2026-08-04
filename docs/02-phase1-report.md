# Phase 1 汇报：Vision MVP

> IndustrialVision-QC | 日期 2026-08-04 | 仅做 Vision MVP，不进入 Phase 2。

## 1. 环境

| 项 | 版本 / 值 |
|---|---|
| GPU | NVIDIA RTX 5060, 8 GB, compute capability (12, 0) 即 sm_120 |
| CUDA 冒烟测试 | 通过（GPU matmul + ultralytics GPU forward） |
| Python | 3.11.9（系统） |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| ultralytics | 8.4.115 |
| opencv-python | 4.12.0.88（5.x 与 torch 在 Windows 上有 DLL 冲突，已降级） |
| numpy | 2.2.6 |
| pydantic | 2.13.4 |
| pytest | 9.1.1 |

依赖在 CUDA 冒烟测试通过后才冻结写入 requirements.txt。

## 2. 数据集

| 项 | 值 |
|---|---|
| 名称 | NEU-DET（热轧钢带表面缺陷） |
| 来源 | GitHub 镜像 `DISHENGRZH/NEU-DET-Steel-Surface-Defect-Detection`（sparse checkout 仅拉取数据文件夹；原始数据集出自东北大学 Song & Yan） |
| 许可 | 研究用途，引用 K. Song and Y. Yan, Applied Surface Science, vol.285, pp.858-864, 2013 |
| 图像 | 1800 张灰度 jpg，200×200 |
| 类别 | crazing / inclusion / patches / pitted_surface / rolled-in_scale / scratches（每类 300） |
| 溯源 | `model-training/datasets/neu-det-yolo/provenance.json` 含 sha256 清单与许可说明 |
| 入库 | 否（`model-training/datasets/` 与 `*.pt` 均在 .gitignore） |

## 3. 数据划分

固定种子 42，按主缺陷类分层划分，每类按 70/15/15 比例分配。

| Split | 数量 | 类别均衡（每类样本数） |
|---|---|---|
| train | 1260 | 203 至 218 |
| val | 271 | 44 至 47 |
| test | 269 | 43 至 47 |

图像级泄漏检查通过（同一文件名不得跨集合，已强制）。

## 4. 模型配置

YOLOv8s（COCO 预训练），在 NEU-DET 上微调。无调参，仅做可靠 baseline。

| 参数 | 值 |
|---|---|
| 模型 | yolov8s |
| imgsz | 256 |
| batch | 32 |
| epochs | 60 |
| patience | 20 |
| seed | 42 |
| workers | 4 |
| device | cuda:0 |

## 5. 训练结果

训练时长 **5.9 分钟**（RTX 5060）。每 epoch 约 4 秒。最佳验证集：

| 指标 | 值 |
|---|---|
| mAP50 | 0.756 |
| mAP50-95 | 0.428 |

混淆矩阵、PR 曲线等已由 ultralytics 自动保存于 `runs/detect/val`。

## 6. 测试集指标（独立评估，269 张图）

| 指标 | 值 |
|---|---|
| Precision | 0.685 |
| Recall | 0.656 |
| mAP50 | 0.708 |
| mAP50-95 | 0.414 |

每类 AP50：

| 类别 | AP50 | AP50-95 |
|---|---|---|
| crazing | 0.307 | 0.110 |
| inclusion | 0.843 | 0.472 |
| patches | 0.827 | 0.535 |
| pitted_surface | 0.844 | 0.522 |
| rolled-in_scale | 0.482 | 0.278 |
| scratches | 0.947 | 0.569 |

crazing 是公认难类（细微网状纹），与公开文献一致。评估完整 JSON 见 `runs/neu-det-yolov8s-baseline/test_metrics.json`。

## 7. 推理性能（RTX 5060, imgsz 256）

10 次 warm-up + 100 次正式测量，test 集图像：

| 指标 | 值 |
|---|---|
| 设备 | cuda:0 |
| 平均延迟 | 11.11 ms |
| P50 延迟 | 11.1 ms |
| P95 延迟 | 12.92 ms |
| 吞吐 | 90 FPS |
| GPU 显存（分配） | 42.5 MB |
| GPU 显存（保留） | 76.0 MB |

## 8. 标准化 Vision Contract（Phase 1C）

`inference-service/app/vision_contract.py` 通过 Pydantic 2 实现，`extra="forbid"` 强制拒绝 schema 外字段。关键事实：

- `severity`、`quality_result` 字段被 schema 拒绝，测试覆盖
- `inference_latency_ms`、`device`、`timestamp` 必填且验证

无真实模型情况下单测产出符合 schema 的样本；有真实模型情况下集成测试断言 bbox 在图像内、confidence 与 area_ratio 在 [0,1]。

## 9. 新增/修改文件

| 路径 | 说明 |
|---|---|
| `inference-service/app/vision_contract.py` | Pydantic 契约（extra=forbid） |
| `inference-service/app/yolo_predictor.py` | YOLO 封装器，CPU 回退与显式错误 |
| `inference-service/tests/test_vision_contract.py` | 契约 12 测试 |
| `inference-service/tests/test_predictor.py` | 预测器 10 测试（含 2 集成） |
| `inference-service/tests/conftest.py` | 路径注入 |
| `model-training/prepare_dataset.py` | XML→YOLO、分层 split、checksum、泄漏检查 |
| `model-training/train_yolo.py` | ultralytics 训练入口 |
| `model-training/evaluate.py` | 测试集评估 |
| `model-training/benchmark.py` | RTX 5060 性能测试 |
| `model-training/cuda_smoke.py` | 冒烟测试 |
| `model-training/demo_inference.py` | 标注图与契约 JSON 生成 |
| `docs/01-negative-sample-strategy.md` | PASS 来源与指标诚实性 |
| `.gitignore` | datasets/ weights runs/ 已屏蔽 |

## 10. pytest

`22 passed, 0 skipped`：契约 12 + 预测器单测 8 + 集成 2。集成测试验证真实模型输出符合契约，CPU 显式路径可加载。

## 11. Git 状态

- 新提交（截至 Phase 1）：
  - 27657a0 archive
  - 4d83999 finalize archive cleanup
  - 85027f8 drop duplicated test sources
  - 881c7e4 phase 0 docs
  - 604ee5c phase 1 vision pipeline code
  - e4dbb91 phase 1 dataset script fixes
- `git diff --check`：clean
- `git ls-files` 不含 datasets/ 与 weights
- 工作树 clean

## 12. 已知问题

1. **opencv-python 5.x DLL 冲突**：5.0.0.93 与 torch 2.11 在 Windows 上产生访问冲突，必须用 4.12.0.88。
2. **Windows DLL 偶发访问冲突**：pytest 收集阶段 torch 导入偶发崩溃，1 至 2 次重试可恢复。已知 Windows + pytest + torch 现象。
3. **crazing 弱类**：AP50 仅 0.31，与文献一致，需要更大模型或更长训练才能提升。
4. **Docker 未安装**：Phase 2 需要，数据库/Redis/MinIO 编排依赖 Docker Desktop。

## 13. Phase 2 建议

- Phase 2 是后端 MVP，不应重新训练模型。直接复用 `best.pt`。
- 安装 Docker Desktop（你已确认）以启用 PostgreSQL/Redis/MinIO。
- 设计 Quality Rule Engine：`quality_rules` 表 + RuleEngine 类，规则版本化，禁用硬编码阈值。
- Phase 2C 通过推理服务把 YoloPredictor 暴露为 HTTP API，后端通过 HTTP 调用而非进程内 import，保留独立部署边界。
- 负样本策略：用 MVTec AD normal 类作为真实清洁样本验证规则引擎 PASS 路径（已写入 `docs/01-negative-sample-strategy.md`）。
- 注意 Phase 2 期间可能再现 Windows 路径/权限类坑，建议第一次 `docker compose up` 前先验证 Docker 安装与 WSL2。