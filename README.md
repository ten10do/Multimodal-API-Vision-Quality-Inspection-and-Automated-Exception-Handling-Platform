# Vision QC Agent

基于多模态 API 的机器视觉质检与异常闭环自动化平台。

用户上传产品图片后，系统通过阿里云百炼识别缺陷，再由 DeepSeek 完成根因分析、风险评级与处置决策，最后执行模拟放行、人工复检、不良品剔除、工单、通知和停线审批。默认 Mock 模式不需要 API Key，但仍完整经过数据库、Provider Schema、工作流、工具调用、状态流转与审计。

## 快速启动

要求：Docker Desktop 与 Docker Compose。

```bash
docker compose up --build
```

默认使用 `AI_MODE=mock`，不要求 `.env`。启动后访问：

- 操作台：http://localhost:3000
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health
- 就绪检查：http://localhost:8000/ready

需要自定义配置时复制模板：

```bash
cp .env.example .env
```

PowerShell：

```powershell
Copy-Item .env.example .env
```

## 本地开发

后端要求 Python 3.12：

```bash
cd apps/api
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload
```

PowerShell 激活命令为：

```powershell
.venv\Scripts\Activate.ps1
```

前端要求 Node.js 22：

```bash
cd apps/web
npm install
npm run dev
```

## Mock 演示数据

安装后端依赖后，在仓库根目录运行：

```bash
python scripts/create_sample_images.py
```

脚本生成四张确定性 PNG：

- `mock-pass.png`：自动放行
- `mock-medium.png`：人工复检
- `mock-high.png`：剔除并创建工单
- `mock-critical.png`：创建停线申请，等待人工审批

结果取决于图片内容散列，不依赖文件名或前端硬编码。

## 工程验证

```bash
make backend-lint-typecheck
make backend-tests
make frontend-lint-typecheck-tests
make frontend-build
make e2e
```

也可以运行：

```bash
make lint
make test
```

浏览器首次执行前安装 Chromium：

```bash
cd apps/web
npx playwright install chromium
```

## 真实 Provider

真实调用只允许在未提交的 `.env` 中配置：

```dotenv
AI_MODE=real
BAILIAN_API_KEY=
BAILIAN_BASE_URL=
BAILIAN_MODEL=
BAILIAN_TIMEOUT_SECONDS=30
BAILIAN_MAX_RETRIES=2
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=
DEEPSEEK_MODEL=
DEEPSEEK_TIMEOUT_SECONDS=30
DEEPSEEK_MAX_RETRIES=2
```

配置完成后执行：

```bash
make smoke-real-api
```

该命令在 `AI_MODE` 不是 `real` 时直接拒绝运行；每个 Provider 只发起一次最小结构化调用，不输出密钥，也不属于默认 CI。

## 安全原则

- 真实 Key 不写入代码、文档、日志、测试、镜像或 Git。
- `.env`、上传文件、数据库、依赖目录、缓存和构建产物均被忽略。
- Provider 响应经过严格 Pydantic Schema 与置信度归一化。
- Provider 失败时安全降级到人工复检。
- 严重风险只创建停线申请，人工批准前工具层拒绝执行停线。
- CI 固定 `AI_MODE=mock`，不会访问付费 API。

## 项目结构

```text
apps/api            FastAPI、SQLAlchemy、Alembic、Celery、Provider 与工作流
apps/web            Next.js App Router 操作台
packages/shared     跨端契约演进说明
infrastructure      部署扩展预留
scripts             演示数据工具
docs                架构、API、安全与部署说明
sample-data         Mock 演示图片
```

进一步阅读：

- [系统架构](docs/architecture.md)
- [API 与状态机](docs/api.md)
- [安全设计](docs/security.md)
- [部署与验收](docs/deployment.md)
