# 工业视觉 AI 平台成熟度报告

## 1. 当前系统架构

当前链路已形成 Camera → Edge Runtime → D3 inference → Decision → PLC → MES → Human Review → Monitoring 的工业闭环。本次新增 Deployment Documentation、Model Governance 和 Business ROI 三个外围层；它们不进入 D3 特征提取、bank、whitening 或 threshold 计算路径。

治理状态遵循 Development → Validated → Candidate → Production → Retired，并通过 artifact 存在性、SHA-256 与必需 metrics 实施 fail-closed。Operations Dashboard 只读展示版本、状态、hash、回滚、指标和审批历史。

## 2. 工业能力矩阵

| 能力 | 状态 | 证据/说明 |
|---|---|---|
| Vision AI | 完成 | 冻结 D3 DINOv2 ViT-B/14 + ZCA |
| Camera Adapter | 完成 | 工业采集接口与失败语义 |
| Decision / PLC / MES | 完成 | PASS/REJECT/HOLD 闭环 |
| Human Review | 完成 | 不覆盖 AI 原始结果的追加裁决 |
| Edge Runtime | 完成 | 容器运行与资源健康 |
| Monitoring | 完成 | Runtime 与 Drift Warning/Critical |
| Deployment Documentation | 新增 | 需求、架构、SOP、维护指南 |
| MLOps Lifecycle | 新增 | 状态、hash/metric 门禁、历史、回滚 |
| Change Management | 新增 | 审批、职责分离、禁止现场替换 |
| Business ROI | 新增 | 可复算的 simulation assumption |
| Production deployment | 未授权 | 本次不部署生产 |

## 3. 与真实工业系统的差距

- Camera、PLC、MES 的真实厂商协议、冗余网络、时钟同步和现场电气安全仍需站点集成验证。
- SLA、容量、备份恢复和灾备目前是设计/模拟证据，尚无连续生产月数据。
- 治理历史使用本地 JSON 审计；大规模多节点部署需迁移到具备签名、RBAC、不可篡改存储和并发控制的注册中心。
- ROI 数字均为 simulation assumption，需由真实工时、缺陷逃逸和质量成本校准。
- 生产网络安全、账号生命周期、漏洞管理和法规/客户审计仍需现场责任方签署。

## 4. 下一步建议

1. 在不改冻结模型的前提下完成客户站点网络、Camera、PLC/MES 接口联调和安全评审。
2. 运行 shadow/pilot，收集至少四周 SLA、HOLD、人工复核、漂移和业务成本基线。
3. 将本地治理日志接入企业制品库、身份系统和不可篡改审计平台。
4. 依据现场数据更新 ROI，完成 FAT/SAT 差异关闭后再申请生产部署授权。

## 5. 交付边界与审计基线

本次仅新增外围工程能力，禁止并未执行 retraining、fine-tuning、threshold tuning、生产部署或 production candidate manifest 修改。实施前 HEAD 为 `ee7888f1ac3758ce05f085b5edb55c8f2b675864`；weights、whitening、image bank、R-L1、R-L2 与 candidate manifest 的 SHA-256 均与 release manifest 匹配。最终交付需再次执行相同校验并以 Git diff 证明冻结 D3 未变化。

## 6. 验证结果

- 新增交付测试：76 passed，覆盖文档、生命周期、fail-closed、回滚、ROI、Dashboard API 与端到端故障回滚。
- 仓库默认全量回归：630 passed、1 skipped、27 deselected；默认配置排除需要 live services、GPU、OPC UA 或现场工业环境的测试。
- 已知非阻断项：现有 FastAPI TestClient 产生 1 条依赖弃用 warning。
