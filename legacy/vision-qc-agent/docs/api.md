# API 与状态机

API 默认地址为 `http://localhost:8000`，交互文档位于 `/docs`。

## 系统端点

| 方法 | 路径      | 说明                     |
| ---- | --------- | ------------------------ |
| GET  | `/health` | 进程存活检查             |
| GET  | `/ready`  | 数据库及必要队列依赖检查 |

版本化端点同时提供 `/api/v1/health` 与 `/api/v1/ready`。

## 质检端点

| 方法 | 路径                                | 说明                                           |
| ---- | ----------------------------------- | ---------------------------------------------- |
| POST | `/api/v1/inspections`               | 上传图片并创建质检任务，要求 `Idempotency-Key` |
| GET  | `/api/v1/inspections`               | 查询质检任务                                   |
| GET  | `/api/v1/inspections/{id}`          | 查询模型、动作、审计与人工反馈                 |
| POST | `/api/v1/inspections/{id}/approval` | 批准或拒绝停线申请                             |
| POST | `/api/v1/inspections/{id}/feedback` | 保存人工复核结果                               |
| GET  | `/api/v1/dashboard/stats`           | 查询仪表盘统计                                 |

错误响应统一包含可追踪的 `request_id`：

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed",
    "request_id": "..."
  }
}
```

响应头同时返回 `X-Request-ID`。

## 状态机

主要状态路径：

```text
queued
  -> vision_analyzing
  -> reasoning
  -> executing
  -> completed | manual_review | awaiting_approval
```

`awaiting_approval` 只能由人工审批进入 `completed` 或 `manual_review`。工具层在停线申请未批准时拒绝执行停线。
