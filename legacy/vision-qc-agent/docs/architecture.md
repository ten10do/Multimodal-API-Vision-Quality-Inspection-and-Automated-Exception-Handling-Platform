# 架构说明

## 边界与数据流

```mermaid
flowchart LR
  U[操作员] -->|上传图片| API[FastAPI]
  API --> V[文件安全校验]
  V --> DB[(PostgreSQL)]
  API --> Q[Redis / Celery]
  Q --> W[持久化工作流]
  W --> VP[VisionProvider]
  VP --> B[阿里云百炼或 Mock]
  W --> RP[ReasoningProvider]
  RP --> D[DeepSeek 或 Mock]
  W --> T[模拟工具执行器]
  T --> DB
  T --> A{停线?}
  A -->|否| C[闭环完成]
  A -->|是| H[人工审批]
  H --> C
```

`VisionProvider` 与 `ReasoningProvider` 是业务层唯一可见的 AI 接口。真实实现使用 `httpx` 调用 OpenAI-compatible HTTP API，不依赖任何厂商 SDK；Mock 实现走同一套 Schema、数据库和工作流。

## 状态机

```mermaid
stateDiagram-v2
  [*] --> queued
  queued --> vision_analyzing
  vision_analyzing --> reasoning
  reasoning --> executing
  executing --> completed: 放行或剔除
  executing --> manual_review: 中风险
  executing --> awaiting_approval: 严重风险
  awaiting_approval --> completed: 批准停线
  awaiting_approval --> manual_review: 拒绝停线
  vision_analyzing --> manual_review: Provider 安全降级
  reasoning --> manual_review: Provider 安全降级
```

## Celery 取舍

关键检测不使用 FastAPI `BackgroundTasks`。生产/Docker 模式提交到 Redis-backed Celery，开启 late acknowledgement，并通过数据库任务状态和工具动作幂等键承受重复投递。开发与 CI 的 `CELERY_TASK_ALWAYS_EAGER=true` 只改变执行时机，不绕过数据库、Provider、工作流或审计，因此无需 Redis 也可完整演示。

## 文件存储

当前实现把随机命名的图片保存到受控目录，Docker 使用独立持久卷。数据库只保存元数据和 SHA-256。该设计适合单站点演示；多节点生产环境应将 `validate_and_store_image` 后端替换为对象存储，同时保留随机对象键、内容校验和数据库元数据。
