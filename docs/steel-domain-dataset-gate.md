# IndustrialVision-QC v1.1 Optimization 1：Steel-domain PatchCore Validation

## Dataset Gate 审计报告

状态：**Dataset Gate 完成，数据集未下载，未训练**

日期：2026-08-10
分支：feat/steel-domain-patchcore-v1.1
基线：main @ 0d72347e29f050b8f56f7481022eafb64831b4f4（working tree clean）

---

## 0. 执行边界

本轮仅执行第 1~4 项（Git 保护、数据集调研、审计、推荐）。下载、preprocessing、训练、评估均未启动，等待数据集批准后继续。

---

## 1. Git 保护检查结果

| 检查项 | 结果 |
| --- | --- |
| 当前分支 | feat/steel-domain-patchcore-v1.1（由 main 创建） |
| working tree | clean |
| git diff --check | 无输出 |
| HEAD | 0d72347e29f050b8f56f7481022eafb64831b4f4 |
| origin/main | 0d72347e29f050b8f56f7481022eafb64831b4f4（一致） |
| 最近提交 | 0d72347 ci / 61e1d45 ci / 22897ac fix / 6cd87cd docs / 1fc9023 docs（Phase 10 交付链） |

确认 local main 与 origin/main 一致，v1.0 不受污染。未执行 reset、force push、git clean。

---

## 2. 候选数据集审计

### Candidate 1：Severstal Steel Defect Detection

| 字段 | 内容 |
| --- | --- |
| name | Severstal: Steel Defect Detection |
| source | Kaggle Competition（2019），Severstal PAO（俄罗斯钢厂）发布 |
| license | 无标准开源协议，受 Kaggle 竞赛条款约束；Dataset Ninja 引用官方表述为数据归 PAO Severstal 所有，除适用法律外无使用限制 |
| industrial domain | 热轧扁钢带（flat sheet steel）产线高频相机在线采集 |
| material | 钢材（steel surface） |
| normal sample count | train 12,568 张中 5,902 张无缺陷（约 47%）；全集 18,074 张中 11,408 张无标注（约 63%） |
| anomaly sample count | train 中 6,666 张含缺陷（约 53%） |
| anomaly types | 4 类（ClassId 1~4）：rolled-in scale（氧化皮压入）、patches（斑块）、crazing（网状裂纹）、pitted surface（点蚀） |
| image resolution | 256 × 1600 灰度长条图（存储为 3 通道） |
| image-level labels | 可由 mask 推导（有任一缺陷 mask 即 anomaly） |
| pixel masks | 有，train.csv 中 RLE 编码像素级 mask，可可靠转换为 binary mask |
| dataset size | train 12,568 + test 5,506（test 无公开 label）；原始 zip 估计 4~5 GB |
| download requirements | Kaggle 账号 + 接受竞赛规则 + kaggle API credential（kaggle.json） |
| suitability for PatchCore | 高。真实 normal 样本充足（5,902），memory bank 可只学 normal；像素 mask 支持 Image/Pixel AUROC 与 AUPRO 三指标 |
| domain similarity with NEU-DET | 高。两者均为热轧钢带表面，缺陷类型高度重合（rolled-in scale、patches、pitted surface 在 NEU-DET 中同名出现） |
| limitations | 真实产线光照不均、背景复杂；缺陷细小（单缺陷面积常小于图像 1%）；train/test 分布漂移被竞赛参与者广泛报告；test 无 ground truth，须自建 hold-out split；部分缺陷标注存在歧义 |

### Candidate 2：GC10-DET

| 字段 | 内容 |
| --- | --- |
| name | GC10-DET（Deep Metallic Surface Defect Detection benchmark） |
| source | 论文数据集（Lv et al., Sensors 2020），官方 GitHub 发布，原始下载为百度网盘 |
| license | 官方未声明 license；Roboflow 镜像标注 CC BY 4.0 |
| industrial domain | 钢厂实际产线采集的钢带（steel sheet）表面 |
| material | 钢材 |
| normal sample count | 0。全部图像均为缺陷样本 |
| anomaly sample count | 3,570 张（镜像版本 2,294 张），10 类缺陷（punching、weld line、crescent gap、water/oil/silk spot、inclusion、rolled pit、crease、waist folding） |
| image resolution | 灰度图（论文未统一声明；镜像标注为线扫图） |
| image-level labels | 有（缺陷类别） |
| pixel masks | 无。仅 bounding box 标注 |
| dataset size | 约 1~2 GB |
| download requirements | 百度网盘（国内访问）；Roboflow 镜像需账号 |
| suitability for PatchCore | 低。无 normal 样本，无法构建 normal-only memory bank，与 PatchCore 前提冲突 |
| domain similarity with NEU-DET | 中。同为钢带表面，但缺陷种类（如 weld line、punching）与 NEU-DET 差异大 |
| limitations | 缺 normal 类；无像素 mask，无法计算 Pixel AUROC / AUPRO；原始下载渠道不便 |

### Candidate 3：KolektorSDD2（Kolektor Surface-Defect Dataset 2）

| 字段 | 内容 |
| --- | --- |
| name | KolektorSDD2 |
| source | 卢布尔雅那大学 Vicos Lab（2021），工业伙伴 Kolektor Group d.o.o. 提供 |
| license | CC BY-NC-SA 4.0 |
| industrial domain | 电子换向器（electrical commutator）生产件表面，受控环境下视觉检测系统采集 |
| material | 电子元件表面（非钢材） |
| normal sample count | 2,979（train 2,085 + test 894） |
| anomaly sample count | 356（train 246 + test 110） |
| anomaly types | 划痕、斑点、表面瑕疵等（单类 defect，形状尺寸颜色多样） |
| image resolution | 高分辨率彩色图（官方约 2300 × 1260；部分第三方页面标注 230 × 630，来源存在冲突，需以官方发布页为准） |
| image-level labels | 有（binary good/anomalous，可由 mask 推导） |
| pixel masks | 有，细粒度分割 mask |
| dataset size | 约 0.9~1 GB |
| download requirements | Vicos 官网发布页申请，需同意条款（部分镜像需手动审核） |
| suitability for PatchCore | 中。方法上完全适配（normal/anomaly 齐备 + 像素 mask），且是公开 AD 榜单基准（最高 image AUROC 97.4） |
| domain similarity with NEU-DET | 低。非钢材表面 |
| limitations | 领域不匹配。可作为方法可行性参照，无法支撑 steel-domain validity 论证 |

### 附：NEU-DET 评估（现有 YOLO 训练数据）

| 字段 | 内容 |
| --- | --- |
| name | NEU Surface Defect Database（NEU-DET） |
| source | 东北大学，公开数据集（Kaggle 等多处镜像） |
| 现状 | 本项目 YOLO known-defect 检测已使用 |
| normal sample count | 0。1,800 张全部为缺陷样本，6 类各 300 张（crazing、inclusion、patches、pitted surface、rolled-in scale、scratches），200 × 200 灰度 |
| 结论 | 不具备 normal 类，无法作为 PatchCore normal-only memory bank 的训练来源。保留为 YOLO 通道数据与 domain 参照 |

---

## 3. 推荐结论

**推荐：Severstal Steel Defect Detection（train 子集，自建 hold-out split）**

### 为什么比 MVTec bottle 更适合本项目

1. 领域匹配。MVTec bottle 是玻璃瓶外观检测，与钢材质检无 domain 关联；Severstal 是热轧扁钢带产线实拍，与工业视觉质检场景一致。
2. 数据真实性。MVTec 为实验室受控拍摄，缺陷清晰、分布理想；Severstal 为产线高频相机采集，光照不均、缺陷细小、分布漂移，更接近真实部署条件。
3. 与现有 YOLO 通道同域。NEU-DET（热轧钢带）与 Severstal（热轧扁钢带）共享缺陷类型学，Fusion 论证存在真实的 domain 连接点。

### 为什么适合 PatchCore

1. 真实 normal 充足。5,902 张无缺陷图像可支撑 memory bank 只学 normal，符合 PatchCore 冷启动前提（无缺陷样本富余、缺陷稀缺且不可预测）。
2. 像素 mask 完整。RLE 可可靠转 binary mask，Image-level AUROC、Pixel-level AUROC、AUPRO 三指标均可评估。
3. 方法一致性。保持 Phase 6 配置（WideResNet-50-2、layer2+layer3 拼接 1536 维、随机采样 50,000 patches、固定种子、threshold 取 train normal max、输入 224×224），仅替换数据域，减少变量。

### Normal 定义

RLE 标注全空（四类 ClassId 均无 EncodedPixels）的图像。即产线判定无任何可见缺陷的表面样本。

### Anomaly 定义

至少存在一个非空缺陷 mask 的图像（ClassId 1~4 任意），与 mask 面积和类别无关。

### 与 NEU-DET 的相同与不同

相同：同为热轧钢带表面灰度成像；缺陷类型高度重合（rolled-in scale、patches、pitted surface 在两数据集中同名）；采集均为产线实拍。

不同：NEU-DET 为 200×200 裁剪块、每图单类、全缺陷无 normal、图像级/bbox 标签；Severstal 为 256×1600 长条原图、单图可多类共存、含真实 normal、像素级 mask。二者标签粒度与场景粒度不同，不可直接混用。

### 边界声明（critical）

same steel domain 不等于 identical production distribution。Severstal 数据仅代表 Severstal 产线的高频相机采集分布。即使新 baseline 验证通过，也禁止声称其代表所有真实钢厂生产线。后续 Fusion demo 若将 NEU-DET 图像与 Severstal 图像同框，必须标记为 cross-dataset integration demonstration。

---

## 4. Access、License 与合规

| 项 | 说明 |
| --- | --- |
| Access | 需 Kaggle 账号，接受竞赛规则，kaggle API credential（kaggle.json）。test split 无公开 label，仅下载 train 12,568 张即可 |
| License | 无标准开源协议（Kaggle 竞赛条款 + PAO Severstal 所有权）。数据集本体不入 Git，仅提交 provenance 与 manifest |
| 现状 | **尚未获得数据。** 不伪造已下载状态。需要用户确认后提供 Kaggle 凭据或由用户手动下载 |

---

## 5. 预期指标（诚实区间，非承诺）

| 指标 | MVTec bottle baseline（Phase 6 实测） | Severstal 预期 |
| --- | --- | --- |
| Image-level AUROC | 1.000 | 0.80~0.95 区间（真实产线，缺陷细小、光照不均） |
| Pixel-level AUROC | 0.986 | 显著低于 MVTec，估计 0.70~0.90 |
| AUPRO | 0.955 | 同上，mask 边界模糊会压低数值 |

Severstal 上无公开 PatchCore AD 基准（codesota 榜单中 Severstal 仅有分割类结果），本项属新验证。第一轮 baseline 目的在测量，预设区间仅作实验设计参考，最终以实测为准。

---

## 6. 主要风险

| 风险 | 等级 | 缓解 |
| --- | --- | --- |
| 产线分布漂移，normal 内部方差大，导致 high FP | 高 | 阈值策略记录分数分布；报告 normal FP rate，不只看 AUROC |
| 缺陷细小（常 <1% 面积），Pixel 指标被拉低 | 中 | tile 处理保持分辨率（256×1600 切 256×256），不整图 resize |
| 部分标注存在歧义或漏标（论文指出） | 中 | 对 FP/FN 做人工 failure case 分析，区分 annotation 问题与算法问题 |
| License 非标准开源，作品集公开受限 | 中 | 数据本体不入 Git；报告与代码仅引用元信息 |
| 下载依赖 Kaggle 凭据，外部依赖 | 低 | 需用户确认；确认前不推进 |

---

## 7. 磁盘与时间估计（RTX 5060 8GB，WRN-50-2）

| 项 | 估计 |
| --- | --- |
| 原始下载（train 12,568 张 zip） | 约 4~5 GB（Voxel51 HF 版 1.76 GB，Supervisely 版 1.57 GB，原始 zip 更大） |
| 解压 + tile 处理后工作副本 | 约 8~12 GB（含 6× tiles、特征缓存） |
| 特征提取（约 60,000 tiles，224×224） | 约 2~3 小时（GPU forward 约 0.1~0.2 s/tile） |
| coreset/随机采样 + 全量评估 | 约 1~2 小时 |
| 报告、failure case、注册 | 约 1 小时 |
| 单轮总计 | 约 4~6 小时，单个工作日可完成 |

以上为估计值，未开始下载，实际以执行为准。

---

## 8. 批判性审视（辩证视角）

1. Severstal 是最优可行解，但存在结构性局限。其 normal 定义依赖"标注者未标出缺陷"，真实产线中可能存在漏标缺陷混入 normal 集合，污染 memory bank。这是所有产线数据集共有的问题，PatchCore 对此类污染缺乏原生防御。validation 通过仅证明"该数据分布下可区分"，不应外推为通用钢材异常检测能力。
2. 评估口径需要防御性设计。单图多缺陷共存、缺陷占比极小的情况下，image-level AUROC 可能虚高（大面积缺陷易于识别），pixel-level 指标更能反映定位质量。最终以操作点指标（FP rate、recall、F1）为主报告，AUROC 为辅。
3. 与 Phase 6 的对比要防止误导。MVTec bottle 的 1.000 AUROC 是实验室受控条件下的上限表现，Severstal 数字更低在预期内，较低数值本身即代表更真实的工业条件，对比重点在 domain validity，不在数字高低。

---

## 9. 待确认事项

1. 是否批准 Severstal Steel Defect Detection 作为 steel-domain PatchCore 数据集。
2. Kaggle 凭据获取方式（用户提供 kaggle.json，或用户手动下载后放入约定目录）。
3. 是否接受 tile 预处理策略（256×1600 → 6× 256×256 tiles）作为与 Phase 6 的既定差异。

批准前不执行下载、不训练、不评估。

---

## 10. 批准后执行计划（第 5~19 项，暂不执行）

1. Dataset Provenance：model-training/datasets/severstal-steel/provenance.json，含 download date、split seed、文件计数、SHA256 manifest。数据本体不入 Git。
2. Leakage Test：train_normal paths ∩ test paths = 0 自动化断言；同源工件防护按 ImageId 隔离。
3. Split：train_normal / validation_normal / test_normal / test_anomaly，anomaly 严禁进入 memory bank。
4. Steel PatchCore：独立 identity steel-patchcore v1.0.0，记录 backbone、layers、input、sampling、memory bank size、threshold、dataset version、artifact SHA256。不覆盖 mvtec-bottle-patchcore。
5. Baseline 单轮：不做调优，保持 Phase 6 方法一致。
6. Evaluation：Image AUROC、Pixel AUROC、AUPRO（若 mask 可靠），外加操作点 FP rate、anomaly recall、precision、F1、confusion matrix。
7. Score Distribution：train/test normal 与 anomaly 分布、threshold、分离度分析。
8. Failure Case：TP/TN/FP/FN + anomaly heatmap，产出 docs/steel-patchcore-failure-analysis.md。
9. 与 MVTec 对比表：Domain、Dataset、各指标、FP rate、latency、memory bank，不做跨数据集"好坏"判断。
10. Fusion 接入：YOLO + PatchCore → Fusion，保持四状态，标注 cross-dataset 边界。
11. MLOps：Model Registry、Dataset Version、Artifact SHA256、MLflow run、Promotion Gate、Deployment Manifest；steel 通过门禁则 domain_validated=true，MVTec 保持 domain_validated=false 且不删除。
12. Promotion 不自动执行，先 CANDIDATE。
13. Tests：provenance、normal-only bank、zero overlap、manifest hash、anomaly contract、threshold boundary、steel normal/anomaly、registry、dataset version、domain validated、promotion gate、Fusion 与 Human Review 回归。
14. Git：大文件不入库，可提交 provenance/manifest/metrics JSON/failure-case 小图/docs/scripts/tests；保持 diff clean、tests green，不 merge main。

---

## Sources（检索来源，均为非中文信源）

- Kaggle Competition：Severstal: Steel Defect Detection，https://www.kaggle.com/competitions/severstal-steel-defect-detection
- Voxel51 / Hugging Face Dataset Card：severstal_steel_defects（18,074 张、11,408 无标注、train 12,568 + test 5,506），https://huggingface.co/datasets/Voxel51/severstal_steel_defects
- Frontiers：Surface Defect Segmentation Algorithm of Steel Plate Based on Geometric Median Filter Pruning（train 12,568 中 5,902 无缺陷 / 6,666 有缺陷，256×1600），https://www.frontiersin.org/articles/945248
- Dataset Ninja：Severstal 统计与下载说明，http://datasetninja.com/severstal
- MDPI Sensors：Lv et al., Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network（NEU-DET 1,800 张 6 类、GC10-DET 3,570 张 10 类），https://www.mdpi.com/1424-8220/20/6/1562/htm
- Roboflow / HuggingFace 镜像：imaadd05/gc10-det（CC BY 4.0，bbox COCO），https://huggingface.co/datasets/imaadd05/gc10-det
- arXiv 2104.06064：Božič et al., Mixed supervision for surface-defect detection（KSDD2 划分与标注说明），https://arxiv.org/abs/2104.06064
- Dataset Ninja：KolektorSDD2 统计，http://datasetninja.com/kolektor-surface-defect-dataset-2
- MDPI Applied Sciences 16(6):3022：Cross-Dataset Benchmarking of Deep Learning Models for Surface Defect Classification in Metal Parts（X-SDD、KSDD2、NEU-DET、DAGM 对比），https://www.mdpi.com/2076-3417/16/6/3022/xml
- Emergent Mind：PatchCore 方法与 MVTec 基准（image AUROC 99.1%、pixel 98.1%、PRO 93.5%），https://www.emergentmind.com/topics/patchcore
- Codesota Anomaly Detection 榜单（KSDD2 最高 97.4、Severstal 无 AD 结果），https://codesota.com/tasks/anomaly-detection
- 项目内部：docs/09-phase6-report.md（MVTec bottle baseline 配置与指标）、inference-service/scripts/train_patchcore.py、inference-service/scripts/benchmark_phase6.py
