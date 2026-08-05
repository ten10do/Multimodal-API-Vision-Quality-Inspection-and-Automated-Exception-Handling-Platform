# Phase 6 Unknown Anomaly Detection

PatchCore 异常检测 + YOLO/PatchCore Vision Fusion。本阶段的核心边界声明：

**MVTec benchmark performance ≠ NEU-DET production-domain performance。**

Image AUROC = 1.000 / Pixel AUROC = 0.986 / AUPRO = 0.955 仅代表 PatchCore
算法在 MVTec AD `bottle` 单类上的能力，**不构成**对钢材质检系统异常识别
准确率的声明。钢材域（NEU-DET）在本阶段只有架构连通性 demo，没有任何
量化指标背书。

## 数据集（docs/08-anomaly-dataset-strategy.md）

- Benchmark：MVTec AD `bottle`（HuggingFace 镜像 `TheoM55/mvtec_all_objects_split`）。
  train 209 normal / test 83（63 anomaly）/ mask 63。
- Industrial-domain：NEU-DET 钢材（无 normal 类），仅用于 Fusion 架构
  连通性 demo，明确标注为跨域受控样本。
- 官方 MVTec mydrive 链接 2025 年末起不稳定，仓库使用 `scripts/fetch_mvtec_bottle.py`
  从 HuggingFace parquet 直接物化标准目录结构。

## PatchCore Baseline（6A/6C，可复现）

```
scripts/fetch_mvtec_bottle.py    # 下载数据
inference-service/scripts/train_patchcore.py   # 训练 bank（209 normal -> 50000 patches）
inference-service/scripts/eval_patchcore.py    # 评估 + 示例图
```

| 指标 | 值 |
|---|---|
| Image-level AUROC | 1.000 |
| Pixel-level AUROC | 0.9863 |
| PRO / AUPRO | 0.9545 |
| threshold | 0.2027（= train normal max，训练集 0 FP） |
| normal score 分布 | 0.0925 - 0.2223（mean 0.1381） |
| anomaly score 分布 | 0.2281 - 0.5307（mean 0.4081） |
| latency（模型级，GPU） | mean 755ms / p50 762ms / p95 791ms |
| peak VRAM（PatchCore 单独） | 289.5 MB |

架构：WideResNet-50-2（torchvision，ImageNet 预训练）提取 layer2+layer3
特征，layer3 上采样到 layer2 网格后 channel 拼接（1536 维），memory bank
随机采样 50000 patches（固定种子，论文级 greedy coreset 留待后续），
patch 最近邻余弦距离生成 anomaly map，图像级得分取 map 最大值。

### Leakage sanity check（已记录证据，未重训）

- train/test 文件路径零重叠（含 ground-truth masks：146 test 文件 vs 209 train）。
- bank 记录 `train_images=209`，与 train/good 实际数量一致；bank 为纯
  patch 特征，无任何 mask/GT 信息。
- threshold 0.2027 等于训练时记录的 train normal max（拟合只使用 normal
  training 数据）。

### 示例图（docs/patchcore-eval/）

- MVTec 评估：`normal.png`、`anomaly-broken_large.png`、
  `anomaly-broken_small.png`、`anomaly-contamination.png`、
  `false_positive.png`（bottle good 005/006，score 0.2156/0.2223）、
  `false_negative.png`。
- 真实链路：`real-known-defect.png`、`real-unknown-anomaly.png`、
  `real-known-defect-with-anomaly.png`。

## Anomaly Contract（6D）

`AnomalyResult`（vision_contract，`extra="forbid"`）：

```
model_name / model_version / anomaly_score / threshold / is_anomalous
regions[]: bbox_xyxy / bbox_normalized / area_ratio / region_score
latency_ms / anomaly_map_png (base64 PNG heatmap，Review UI 使用)
```

Contract 强制无质量判断字段（PASS/REVIEW/FAIL/severity 均被拒绝）。
`is_anomalous` 仅是模型阈值判断，不构成业务 FAIL。

## Fusion Layer（6E/6H）

`inference-app/fusion.py` 纯函数，四状态：

| YOLO detections | PatchCore | fusion_class |
|---|---|---|
| 0 | normal | NORMAL_CANDIDATE |
| >0 | normal | KNOWN_DEFECT |
| 0 | anomaly | UNKNOWN_ANOMALY |
| >0 | anomaly | KNOWN_DEFECT_WITH_ANOMALY |

PatchCore 不可用（bank 缺失/加载失败/推理异常）时 anomaly=None，
fusion 回退为 YOLO-only 视图（不产生虚假 UNKNOWN_ANOMALY）。Backend
只经 HTTP 接收 `VisionResult`（detections + anomaly + fusion_class +
各阶段延迟），不 import 任何视觉模型（6H）。

## Quality Rule Integration（6F）

`QualityRuleEngine.evaluate(detections, fusion_class)`：

- `UNKNOWN_ANOMALY` → REVIEW（PatchCore 只说明偏离 normal 分布，不证明
  工艺不合格；第一版不做 UNKNOWN_ANOMALY → FAIL）。
- 其余 fusion_class 走既有 per-detection 规则（KNOWN_DEFECT_WITH_ANOMALY
  不因 anomaly 通道强制 FAIL）。

## Human Review Integration（6G）

- ReviewTask 快照新增 `anomaly_score / anomaly_threshold / is_anomalous /
  anomaly_regions / anomaly_map_url`。
- Review UI 详情：Anomaly Heatmap（`GET /inspections/{id}/anomaly-map` PNG）、
  score / threshold / regions 表。
- UNKNOWN_ANOMALY 任务（无 YOLO 缺陷）决策限制为
  PASS / CONFIRM_DEFECT / OTHER_DEFECT（无 AI label 可纠正，隐藏
  CORRECT_DEFECT）。
- `TrainingCandidate` 新增 `anomaly_score`；UNKNOWN_ANOMALY 人工确认后，
  human_label + image + anomaly score 进入候选（unknown → known 闭环，
  不自动 retrain）。

## 真实 Fusion 分布解释（阈值合理性）

真实跨域 demo（NEU 图 × bottle-normal bank）分布：

```
KNOWN_DEFECT_WITH_ANOMALY = 359（真实链路）
UNKNOWN_ANOMALY           = 41（真实链路）
KNOWN_DEFECT              = 受控样本（bottle good 图被 NEU-YOLO 误检）
NORMAL_CANDIDATE          = integration fixture（跨域数据无法自然产生）
```

大量已知缺陷同时是 PatchCore anomaly 是**预期行为**，不是重复检测 bug：
NEU 钢材图像相对 bottle-normal 特征库分布偏移极大（抽样 10 张 NEU 图
score 0.517-0.556，100% ≥ threshold 0.2027）。这是 anomaly detector 对
"偏离 normal distribution"的响应；YOLO 已知缺陷与 PatchCore 异常是两条
独立通道。阈值在拟合域（bottle）上贴合边界（normal max 0.2223，仅 2/20
正常图略超），不构成"阈值过低导致所有图像 anomalous"。

受控样本均来自真实服务链路（真实 YOLO + PatchCore + fusion），只是输入
被选择/标记为 integration fixture，不代表 benchmark 分布。

## 性能（6I，docs/phase6-benchmark.json）

模型级（同一进程，RTX 5060 8GB，12 样本 + 2 warmup）：

| 阶段 | mean | p50 | p95 |
|---|---|---|---|
| YOLO | 15.4ms | 13.3ms | 21.5ms |
| PatchCore | 755.0ms | 761.6ms | 790.6ms |
| Fusion | 0.0ms | 0.0ms | 0.0ms |
| Total Vision | 770.3ms | 776.6ms | 811.8ms |

- Peak GPU allocated 331.3MB / reserved 356.0MB（两模型共存，远低于 8GB）
- Fusion 分布（12 样本）：UNKNOWN_ANOMALY 5 / KNOWN_DEFECT_WITH_ANOMALY 7

生产流水线级（simulator → backend HTTP → inference → PG，500 样本）：

- E2E inference_latency mean 456.2ms / p50 438.8ms / p95 521.2ms
- throughput ≈ 2.1/s（10s 窗口），realtime field 1.967/s

PatchCore 755ms 是 baseline（WRN50 前向 + 784×50000 最近邻）；优化
（coreset 缩库、特征降维、batch）留待后续，本阶段不做复杂 model
scheduling。

## 测试（6J）

- inference：fusion 四状态、threshold 边界、anomaly contract（禁质量字段）、
  VisionResult、PatchCore GPU smoke。
- backend：fusion 规则（UNKNOWN_ANOMALY→REVIEW 等）、review anomaly 集成
  （task 快照 / resolve / training candidate）、NORMAL_CANDIDATE fixture、
  PatchCore 不可用回退。
- integration：Docker PG 真实并发 claim（Phase 6 前置门禁）。
- Browser E2E（13 通过）：UNKNOWN_ANOMALY human review（heatmap + score +
  regions 消费、CONFIRM_DEFECT、CORRECT_DEFECT 隐藏、training candidate）、
  dashboard / review / 409 冲突全量回归。

## 已知问题

1. PatchCore 单图 755ms（baseline）；生产 E2E 456ms 因并发与 batch。
2. NEU 跨域 demo 下几乎所有图像 is_anomalous（domain mismatch 预期）；
   钢材域异常识别能力无量化声明，需真实钢材 normal/异常数据后重新训练
   bank 并评估（见 docs/08）。
3. NORMAL_CANDIDATE 与 KNOWN_DEFECT 在当前跨域数据下无自然样本，由
   integration fixture 验证逻辑链路（已明确标记）。
4. MVTec 下载依赖 HuggingFace 镜像可用性。

Phase 6 完成后停止，不进入 Phase 7。
