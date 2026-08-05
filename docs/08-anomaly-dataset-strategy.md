# Phase 6 Anomaly Dataset Strategy

## 结论先行

- **Benchmark Dataset**：MVTec AD `bottle` 单类，用于验证 PatchCore 算法本身
  的检测能力（Image AUROC 1.0 / Pixel AUROC 0.986 / AUPRO 0.955）。
- **Industrial-domain Dataset**：当前阶段没有公开可用的「钢材表面 + 真实
  normal 样本 + 异常样本」且质量足够的数据集可直接用于本系统领域评估。
  因此按约定执行：
  - Benchmark 完整完成（MVTec bottle）；
  - Integration（与钢材 YOLO Pipeline 融合）使用**明确标注为 demo** 的
    有限样本（NEU-DET 钢材图像 + MVTec-bottle-normal 训练出的 PatchCore，
    属跨域 demo）。
- MVTec benchmark 的指标**绝不**用于为 NEU-DET 钢材系统的 PASS 能力背书；
  钢材域的异常检测能力本阶段**不做任何量化声明**。

## 两类数据的明确区分

| 维度 | Benchmark Dataset | Industrial-domain Dataset |
|---|---|---|
| 来源 | MVTec AD `bottle`（HuggingFace 镜像 `TheoM55/mvtec_all_objects_split`） | NEU-DET 钢材缺陷图像（已有，仅缺陷类，无 normal 类） |
| 领域 | 塑料瓶表面 | 热轧钢材表面 |
| normal 样本 | 209 张（train/good） | 无（NEU-DET 六类全为缺陷） |
| 异常样本 | 63 张（broken_large / broken_small / contamination） | 六类已知缺陷（YOLO 可识别） |
| 像素级标注 | 63 张 mask | 无 |
| 用途 | PatchCore 算法验证（AUROC / AUPRO / 阈值 / 分布） | 仅用于 Vision Fusion 架构连通性演示 |
| 指标声明 | 允许量化 | **不量化**；demo 结果仅证明链路可用 |

## 数据获取与复现

官方 MVTec mydrive 下载链接自 2025 年末起不稳定，本仓库使用 HuggingFace
镜像（`TheoM55/mvtec_all_objects_split`）直接下载 bottle 的 parquet
分片，并物化为标准 MVTec 目录结构：

```
scripts/fetch_mvtec_bottle.py
```

产出：

```
model-training/datasets/mvtec/bottle/
  train/good/           209 张 normal（训练 PatchCore memory bank）
  test/good/            20 张 normal（测试）
  test/broken_large/    20 张（异常）
  test/broken_small/    22 张（异常）
  test/contamination/   21 张（异常）
  ground_truth/<defect>/  63 张像素级 mask
```

## Domain Limitation（诚实声明）

1. PatchCore memory bank 由 MVTec bottle normal 图像构建；
   NEU-DET 钢材图像不属于同一特征分布。
2. 钢材集成 demo 中，PatchCore 对 NEU 图像的异常判定仅用于验证
   Fusion Layer 与 Quality Rule Engine 的连通性。
3. 若未来获得钢材表面真实 normal/异常数据（如 KSDD 类数据集或产线采集），
   应重新训练 bank 并重新评估，本阶段的数字不迁移。
