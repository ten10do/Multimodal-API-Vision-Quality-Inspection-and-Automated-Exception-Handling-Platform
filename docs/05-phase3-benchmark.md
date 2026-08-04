# Phase 3 Benchmark：Realtime Pipeline

> 环境：RTX 5060 (12.0, 8GB) + Docker Desktop PostgreSQL 容器（host 5433）+ 本机原生 PostgreSQL 并行。
> 日期 2026-08-04。配置见下，命令可复现。

## 配置

- 来源：NEU-DET test 目录（200x200 jpg）
- simulator interval：200 ms（固定间隔）
- worker count：2
- queue maxsize：20（block 策略）
- max_images：60
- backend：uvicorn on 8123 → 推理服务 8100（cuda:0）→ 容器 PostgreSQL industrialvision_test
- 重试：retry_max=2，指数退避 300/600ms（本次零重试触发）

## 结果（优化后）

| 指标 | 值 |
|---|---|
| sample count | 60 |
| simulator interval | 200 ms |
| worker count | 2 |
| queue maxsize | 20 |
| peak queue depth | 2 |
| completed / failed | 60 / 0 |
| total duration | 12.38 s |
| throughput | 4.85 inspections/s |
| E2E avg latency | 63.6 ms |
| E2E P50 | 47.4 ms |
| E2E P95 | 73.8 ms |
| inference avg latency | 13.9 ms |
| quality 分布 | PASS 6 / REVIEW 33 / FAIL 21 |

模型延迟与端到端延迟明确分离：inference 13.9ms 只含 GPU 前向 + NMS；E2E 63.6ms 含 HTTP、DB 往返、规则引擎、multipart 序列化与队列节拍。

## 优化记录

首轮 benchmark E2E avg 561.7ms / P95 822ms / 吞吐 3.50/s。定位到 `InferenceClient` 每次请求新建 `httpx.AsyncClient`（Windows 下每次实例化约 150ms，且无连接复用）。修复为按事件循环缓存单个客户端后，E2E avg 降至 63.6ms（约 8.8 倍改善）。该优化同时减少了每个 inspection 的建连开销，未改变任何业务语义。

## 复现命令

```bash
# 准备 test DB（容器内，仅操作 industrialvision_test）
.venv/Scripts/python.exe scripts/prepare_test_db.py --recreate

# 启动后端（指向容器 DB + 推理服务）
cd backend && ../.venv/Scripts/python.exe -m uvicorn app.main:app --port 8123

# 跑 benchmark（需推理服务在 8100 运行）
.venv/Scripts/python.exe -m simulator.run_pipeline \
  --images 60 --interval-ms 200 --workers 2 --queue-size 20 \
  --backend-url http://127.0.0.1:8123 --batch bench-p3-002
```
