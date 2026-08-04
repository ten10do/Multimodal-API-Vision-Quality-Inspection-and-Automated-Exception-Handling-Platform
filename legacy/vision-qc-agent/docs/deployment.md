# 部署与验收

## Docker Compose

```bash
docker compose build
docker compose up -d
docker compose ps
```

验收地址：

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/ready
curl --fail http://localhost:3000
```

停止并删除本地数据卷：

```bash
docker compose down -v
```

## 服务

- `postgres`：质检、模型调用、动作、审计和人工反馈
- `redis`：Celery broker 与结果后端
- `api`：执行迁移后启动 FastAPI
- `worker`：消费可靠异步任务
- `web`：Next.js standalone 运行时

API 与 Web 均配置容器健康检查；Web 等待 API 就绪后启动。镜像构建上下文通过 `.dockerignore` 排除 `.env`、依赖、测试、缓存和上传数据。

## CI 验收

GitHub Actions 分别执行：

- 后端 Ruff 与严格 MyPy
- PostgreSQL/Redis 环境下的 Alembic 和 Pytest 覆盖率
- 前端 ESLint、TypeScript、Vitest 覆盖率与生产构建
- Docker Compose 构建、健康检查和最小持久化闭环
- Chromium Playwright 完整停线审批闭环

Compose Job 失败时输出容器日志，并在任何结果下执行 `docker compose down -v`。

## 生产前补充

当前工程面向本地演示和参考实现。生产部署前还需要接入实际身份认证、角色授权、对象存储、集中式密钥管理、速率限制、恶意文件扫描、TLS 和不可篡改审计存储。
