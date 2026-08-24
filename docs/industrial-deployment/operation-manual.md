# 工业运行操作手册（SOP）

## 1. 开班前与每日巡检

操作员按顺序确认并记录：

1. Runtime 状态为 RUNNING/HEALTHY，Camera、Inference、Decision、PLC、MES 服务无错误。
2. GPU 可见，显存、温度、利用率在站点批准范围；CPU、内存、磁盘无持续告警。
3. `/operations` 的 model version、lifecycle state、artifact hash 与当班发布单一致。
4. Drift 状态为 NORMAL；若 WARNING，确认观察单和复核抽样已生效；CRITICAL 时不得开线。
5. 前一班 PASS/REJECT/HOLD 数量守恒，异常率变化有交接说明。
6. 使用测试件完成 Camera → PLC/MES/Review 链路检查，并保存结果。

## 2. 运行中检查

- 每两小时检查请求量、P95 latency、错误率、GPU/内存、队列深度和 Drift。
- HOLD 激增时先保护生产，不调整 threshold；按故障码分流调查。
- 人工复核只能追加裁决，不覆盖原始 AI 结果。
- 交接班记录未完成 HOLD、开放 MES 工单、告警和所有临时控制措施。

## 3. 异常处理

### Camera Failure

```text
Camera failure → HOLD → 人工检查当前件 → 检查供电/网络/触发/帧质量
→ Camera 健康检查通过 → 测试件验证 → 授权恢复
```

禁止使用上一帧或占位图替代真实生产图像。

### AI Failure

```text
Inference error/timeout → HOLD → 保存 request_id 与错误日志
→ 校验容器、GPU、manifest 和 artifact hash
→ 当前版本无法恢复时执行已批准 rollback → 测试件验证
```

Rollback 不是训练、调参或修改 manifest；目标必须是治理历史中可验证的已批准版本。

### Drift Warning

继续生产并观察；创建调查单，提高人工抽检比例，检查照明、表面状态、相机位置和上游材料变化。不得自动调 threshold 或 retraining。

### Drift Critical

当前件及后续件 HOLD；通知质量、设备和发布负责人。只有确认输入环境恢复，或完成已批准回滚并通过测试件验证后，方可恢复。

## 4. 停线与恢复

停线时先阻止新触发，再排空/持久化队列，确认 PLC 安全状态并保存审计日志。恢复按 Camera → Runtime → Inference → Decision → PLC → MES → Review 顺序验证；任一步失败即停止恢复。
