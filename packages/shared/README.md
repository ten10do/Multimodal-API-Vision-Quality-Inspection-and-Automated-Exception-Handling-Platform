# Shared contracts

后端 Pydantic Schema 与自动生成的 OpenAPI 文档是跨端契约源。前端类型目前保持为小型手写只读投影，避免在首版引入额外代码生成链。进入多人迭代后，可由 `/openapi.json` 生成 TypeScript 类型并在此包发布。
