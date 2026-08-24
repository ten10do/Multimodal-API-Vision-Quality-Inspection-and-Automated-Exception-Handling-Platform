# 工业系统维护指南

## 1. 日常维护

- 每日：检查服务健康、资源、队列、错误率、磁盘余量和 Drift 状态。
- 每周：验证告警通知、日志轮转、备份恢复样本和 PLC/MES 对账。
- 每月：执行只读 artifact 校验、权限复核、回滚演练和审计日志抽查。
- 计划维护前生成变更单；维护后使用测试件完成全链路验证。

## 2. 日志查看

以 `request_id`、`inspection_id`、`product_id`、`camera_id`、`model_version` 为关联键，依次查看 Camera、Runtime、Inference、Decision、PLC/MES 和 Review 日志。日志只用于诊断；不得编辑历史记录来掩盖故障。

优先级：安全/HOLD → artifact 或模型身份 → 控制链 → 性能 → 业务统计。导出日志前清理凭据和个人信息。

## 3. Artifact 校验

1. 从批准的 release manifest 读取 artifact URI 与期望 SHA-256。
2. 确认文件存在、只读且路径在批准的项目/制品根目录内。
3. 计算 SHA-256 并逐项比较 weights、whitening、image bank、R-L1、R-L2 和 candidate manifest。
4. 任一缺失或不匹配：fail closed，禁止启动、晋升或回滚到该版本。
5. 将结果、时间、操作员和 manifest 版本写入维护记录。

不得“修复”冻结 artifact、重建 bank、修改 threshold 或覆盖 manifest；应从批准制品库恢复原文件。

## 4. 回滚流程

1. 创建事件/变更单，记录失败版本、症状、影响和目标版本。
2. 从 `model_history.json` 选择上一已验证版本。
3. 治理管理器验证目标 artifact 存在、SHA-256 匹配且所需 metrics 完整。
4. 执行 rollback；确认失败版本 RETIRED、目标版本 PRODUCTION、rollback status 为 COMPLETED。
5. 重启外围服务仅限现场运行手册授权范围；使用测试件验证后恢复生产。
6. 对账 PLC/MES/Review，并归档日志。禁止现场直接替换模型文件。

## 5. 故障排查速查

| 现象 | 检查 | 动作 |
|---|---|---|
| 无图像 | 供电、触发、网口、camera_id、帧校验 | HOLD，恢复 Camera 后测试 |
| GPU 不可用 | 驱动、容器设备映射、资源监控 | HOLD，恢复运行环境 |
| 推理错误 | request_id、输入契约、服务日志、artifact hash | 校验失败则回滚 |
| PLC 无确认 | 网络、会话、命令幂等键、PLC 状态 | HOLD 并人工隔离当前件 |
| MES 积压 | outbox、重试、去重键、MES 可用性 | 保留事件并受控重放 |
| Drift Warning | 照明、相机、材料、分数/embedding 分布 | 观察与增抽检 |
| Drift Critical | 同上并检查近期批准变更 | HOLD，调查或回滚 |
