# Normal / Negative Sample Strategy

> 版本 v0.1，日期 2026-08-04。本文件回答一个问题：质检链路的 PASS 样本从哪来，以及为什么真实指标不会被合成数据污染。

## 1. 问题定义

NEU-DET 只包含缺陷样本（1800 张，6 类缺陷，每张图至少一个缺陷）。YOLO 在纯缺陷数据上训练，无法验证它在真实无缺陷表面上的行为。生产流水线里 PASS 是常态，负样本（normal / no-defect）的可靠来源必须提前设计，否则会出现两类失真：

- 指标失真：用合成负样本冒充真实场景，得到的假阳性率没有说服力。
- 行为失真：模型在真实清洁表面上频繁误报，规则引擎却无从区分。

## 2. 分层策略

| 阶段 | 负样本来源 | 用途 | 是否计入真实指标 |
|---|---|---|---|
| Phase 1 | 无（NEU-DET 全为缺陷样本） | YOLO 训练与测试集评估均不含无缺陷样本 | 否 |
| Phase 2 | 无 | `detections 为空 → PASS` 仅验证业务链路行为，不构成无缺陷性能证明 | 否 |
| Phase 3+ | 相机模拟器流水线帧，零检测帧进入 PASS 候选 | 累积 normal sample bank | 候选，未复核前不计入 |
| Phase 5+ | 人工复核批准的无缺陷图像 | 构成负样本训练集 | 计入（复核后） |
| PatchCore 阶段 | 具有正常/异常结构的数据集（同领域优先） | 正式异常检测评估 | 计入 |

领域边界约束：NEU-DET 是热轧钢带表面，MVTec AD 是瓶罐/螺丝/坚果等异质物体表面，**领域不同，MVTec AD 的 normal 类不得作为当前钢带 YOLO 模型的 PASS 性能证明**。正式的无缺陷性能评估必须使用同领域真实正常钢材样本。

## 3. 规则引擎的 PASS 定义（Phase 2 起）

PASS 是一个派生结论，由规则引擎给出：

```
PASS  ⇔  无任何检测框同时满足对应缺陷规则的全部阈值
```

推理层只上报 detections 几何事实，不产生 PASS 语义。这一分工在 Phase 1C 契约中已固化（`severity`、`quality_result` 字段被 schema 拒绝）。

## 4. 指标诚实性约束

- 真实检测指标（Precision / Recall / mAP）只在真实 NEU-DET test split 上计算，见 `model-training/evaluate.py`。
- 合成图像只用于确定性单元测试（Phase 1E），测试产物与训练指标在文档与报告中明确分离。
- `detections 为空 → PASS` 在 Phase 2 只表示业务系统行为，不代表模型已证明产品真实无缺陷。
- normal sample bank 中的每张图必须携带来源与复核记录（`provenance.json` 扩展），无来源图像不得进入任何指标集。

## 5. Normal Sample Bank 设计

```
model-training/datasets/normal-bank/
  images/           真实无缺陷图像
  provenance.json   每张图的来源、采集工位、复核人、approved_inspection_id
```

数据约束与 NEU-DET 一致：图像级去重、sha256 清单、禁止混入合成图。该目录与数据集目录同等对待，不入 git。

## 6. 后续重训练路径

当 normal-bank 超过阈值规模（建议每类场景 ≥ 300 张）后，可选的优化路线：

- 作为背景负样本参与训练（增加 `background` 类或引入 Focal 损失），目标降低假阳性。
- 或按 Phase 5 人工修正数据一并进入训练集，由 MLflow 记录数据版本。
