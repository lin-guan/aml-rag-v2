# AML Memory Service

一个可复现、可由 Agent Memory Leaderboard 平台直接构建部署的 Add/Search 长程记忆服务。本项目只负责记忆写入和检索，不生成最终答案，不运行 Judge，不读取评测金标或 evidence 标注。

## 提交和部署方式

本仓库按“代码 + Dockerfile，由评测平台部署”的方式提交。平台构建并启动容器后，服务在容器内监听：

```text
0.0.0.0:8000
```

API 路径：

```text
GET  /health
POST /add
POST /search
```

## 方法概述

1. Add 接收平台发送的消息批次，保留消息角色、内容和可选时间戳；
2. 使用本地 `sentence-transformers/all-mpnet-base-v2` 将记忆文本转换为 L2 归一化向量；
3. 将原文、向量、`user_id`、`session_id` 和 `request_id` 持久化到 SQLite；
4. 使用 `request_id` 保证 Add 幂等，并拒绝同 ID 不同 payload 的冲突写入；
5. Search 只访问请求中指定 `user_id` 的记忆；
6. 使用归一化向量内积进行 Dense 检索，并与 SQLite FTS5 词法检索融合；
7. 可选使用会话窗口索引、邻域扩展、时间记忆表示和 options 查询扩展；
8. 返回最多 `top_k` 条已写入记忆，服务端默认硬上限为 100。

所有增强都位于记忆写入和检索系统内部。服务不会读取答案、Judge 结果或评测标注，也不会生成伪造记忆。

## 模型披露

Add/Search 不调用外部 LLM API。当前实际使用的本地向量模型为：

```text
sentence-transformers/all-mpnet-base-v2
```

它在 Add 阶段编码记忆，在 Search 阶段编码查询。Docker 构建时会下载并固化该模型，运行时通过 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 从镜像缓存加载，以减少平台复现时的网络依赖和模型漂移。

## Docker 构建和运行

### 使用 Dockerfile

平台只需要注入一个参赛密钥；其他运行配置均使用镜像内默认值。

```bash
docker build -t aml-memory-service:latest .
docker run -d \
  --name aml-memory-service \
  --restart unless-stopped \
  -p 8000:8000 \
  -e API_KEY="$MEMORY_SYSTEM_KEY" \
  -e AUTH_MODE=bearer \
  -v aml-memory-data:/service/data \
  aml-memory-service:latest
```

### 使用 Docker Compose

平台可以直接注入 `API_KEY`：

```bash
MEMORY_SYSTEM_KEY="replace-with-a-long-random-secret" \\
  docker compose up -d --build
```

或者在平台的容器环境变量配置中设置：

```text
API_KEY=<platform-injected-secret>
AUTH_MODE=bearer
```

容器会在 `/service/data` 的持久化卷中自动创建空数据库，并由平台后续通过 Add 写入记忆。

检查容器状态：

```bash
curl http://127.0.0.1:8000/health
```

正常响应示例：

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_ready": false
}
```

`model_ready=false` 表示 embedding 模型尚未因 Add/Search 请求而延迟加载，不代表服务异常。第一次 Add/Search 会加载镜像中已缓存的模型。

### 平台部署要求

- 建议 CPU 环境至少 4 核、8 GB 内存；
- 容器必须以单 worker 运行；
- `/service/data` 应挂载持久化卷；
- 容器需要允许读取镜像内 `/models/huggingface` 的模型缓存；
- 健康检查启动宽限期为 120 秒；
- 平台应通过环境变量注入 `API_KEY`，不要把真实密钥写入仓库或镜像；
- Search 的 `top_k` 最大值为 100。

## AML API 契约

### Health

```http
GET /health
```

无需鉴权，任意 2xx 表示服务可用。

### Add

```http
POST /add
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

请求示例：

```json
{
  "request_id": "eval:run_abc123:sample-0:chunk-0",
  "messages": [
    {
      "role": "user",
      "timestamp": 1704067200000,
      "content": "Alice adopted a cat named Luna."
    }
  ],
  "user_id": "eval:run_abc123:sample-0",
  "session_id": "eval:run_abc123:sample-0:session-0"
}
```

成功响应：

```json
{
  "success": true,
  "request_id": "eval:run_abc123:sample-0:chunk-0",
  "user_id": "eval:run_abc123:sample-0",
  "session_id": "eval:run_abc123:sample-0:session-0"
}
```

相同 `request_id` 和相同 payload 重复提交时不会重复写入。相同 `request_id` 对应不同 payload 时返回 HTTP 409。

### Search

```http
POST /search
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

请求示例：

```json
{
  "query": "What is Alice's cat called?",
  "options": ["A. Luna", "B. Milo"],
  "user_id": "eval:run_abc123:sample-0",
  "top_k": 100
}
```

响应示例：

```json
{
  "data": [
    {
      "id": "mem_0123456789abcdef01234567",
      "content": "[2024-01-01T00:00:00Z | user] Alice adopted a cat named Luna.",
      "score": 0.87,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

Search 始终以 `user_id` 隔离数据，不会跨用户返回内容。

## 运行配置

平台只需要注入一个参赛密钥；其他配置使用代码中的默认值：

```text
API_KEY=<platform-injected-secret>
AUTH_MODE=bearer
```

默认运行参数：

```text
DATABASE_PATH=data/memory.db
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
MAX_TOP_K=100
HOST=0.0.0.0
PORT=8000
```

详细参数仍可通过环境变量覆盖，具体配置以 `app/config.py` 为准。

<details>
<summary>可选高级配置（通常无需修改）</summary>

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `API_KEY` | 无 | 非 `none` 鉴权模式下必填，由平台注入 |
| `AUTH_MODE` | `bearer` | `bearer`、`token`、`x_api_key` 或 `none` |
| `DATABASE_PATH` | `data/memory.db` | SQLite 数据库路径 |
| `EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | 本地向量模型 |
| `HF_HUB_OFFLINE` | `1` | 运行时不访问 Hugging Face |
| `TRANSFORMERS_OFFLINE` | `1` | 运行时只读取镜像缓存 |
| `EMBEDDING_DEVICE` | 自动 | 可选 `cpu` 或 `cuda` |
| `EMBEDDING_BATCH_SIZE` | `64` | 编码批大小 |
| `MAX_CONCURRENT_ENCODES` | `1` | 同时编码任务数 |
| `INCLUDE_OPTIONS_IN_QUERY` | `true` | 是否把选择题 options 纳入查询 |
| `MAX_TOP_K` | `100` | Search 返回数量硬上限 |
| `ENABLE_HYBRID_RETRIEVAL` | `true` | 是否启用 FTS5 混合检索 |
| `LEXICAL_CANDIDATE_K` | `100` | 词法候选上限 |
| `DENSE_WEIGHT` | `1.0` | Dense RRF 权重 |
| `LEXICAL_WEIGHT` | `0.9` | 词法 RRF 权重 |
| `NEIGHBORHOOD_RADIUS` | `1` | 邻域扩展范围 |
| `INDEX_WINDOW_ENABLED` | `true` | 是否建立内部会话窗口索引 |
| `INDEX_WINDOW_SIZE` | `6` | 窗口消息数 |
| `INDEX_WINDOW_OVERLAP` | `2` | 窗口重叠消息数 |
| `HOST` | `0.0.0.0` | 容器监听地址 |
| `PORT` | `8000` | 容器监听端口 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

默认配置如下；平台部署时可以通过环境变量覆盖这些值。

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `API_KEY` | 无 | `AUTH_MODE` 不是 `none` 时必须由平台注入 |
| `AUTH_MODE` | `bearer` | 推荐使用 `bearer` |
| `DATABASE_PATH` | `data/memory.db` | 容器内 SQLite 路径，建议挂载 `/service/data` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | Add/Search 使用的本地向量模型 |
| `HF_HUB_OFFLINE` | `1` | 运行时不访问模型仓库 |
| `TRANSFORMERS_OFFLINE` | `1` | 运行时使用镜像内模型缓存 |
| `EMBEDDING_BATCH_SIZE` | `64` | 向量编码批大小 |
| `MAX_CONCURRENT_ENCODES` | `1` | 并行编码任务数 |
| `INCLUDE_OPTIONS_IN_QUERY` | `true` | 是否将 options 拼入 Search 查询 |
| `MAX_TOP_K` | `100` | Search 返回数量上限 |
| `ENABLE_HYBRID_RETRIEVAL` | `true` | 是否启用 FTS5 混合检索 |
| `LEXICAL_CANDIDATE_K` | `100` | 词法候选数 |
| `DENSE_WEIGHT` | `1.0` | Dense 融合权重 |
| `LEXICAL_WEIGHT` | `0.9` | 词法融合权重 |
| `NEIGHBORHOOD_RADIUS` | `1` | 邻域扩展半径 |
| `CONTEXT_EMBEDDING_RADIUS` | `2` | 上下文 embedding 半径 |
| `CONTEXT_EMBEDDING_WEIGHT` | `0.3` | 上下文 embedding 权重 |
| `NEIGHBOR_RESULT_RATIO` | `0.2` | 邻域结果比例 |
| `INDEX_WINDOW_ENABLED` | `true` | 是否启用窗口索引 |
| `INDEX_WINDOW_SIZE` | `6` | 窗口消息数 |
| `INDEX_WINDOW_OVERLAP` | `2` | 窗口重叠消息数 |
| `WINDOW_RETRIEVAL_WEIGHT` | `0.7` | 窗口检索权重 |
| `CODE_RETRIEVAL_ENABLED` | `true` | 是否启用代码 token 检索 |
| `CODE_EXACT_MATCH_WEIGHT` | `0.08` | 代码 token 精确匹配权重 |
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `PORT` | `8000` | 服务端口 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

以上配置说明复制了实际配置文件中的有效默认值，不包含任何密钥。平台通常无需覆盖高级配置。

</details>

## 本地开发和测试

Python 3.10+，建议 Python 3.11。

```bash
python -m venv .venv
```

Linux/macOS：

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest
ruff check .
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest
ruff check .
```

测试覆盖健康检查、鉴权、Add/Search 响应结构、幂等写入、冲突检测、用户隔离、时间记忆和窗口索引。

## 工程结构

```text
.
├── app/
│   ├── config.py
│   ├── embedder.py
│   ├── main.py
│   ├── schemas.py
│   ├── service.py
│   └── store.py
├── scripts/
│   └── healthcheck.py
├── tests/
│   └── test_api.py
├── .dockerignore
├── .env.example
├── compose.yaml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── run.py
└── README.md
```

## 原始工作和依赖披露

- 本仓库提供 AML Add/Search 服务封装、SQLite 存储、请求幂等、鉴权、用户隔离、混合检索、窗口索引、时间记忆和测试；
- 向量编码依赖公开项目 Sentence Transformers 和模型 `sentence-transformers/all-mpnet-base-v2`；
- Web API 使用 FastAPI 和 Uvicorn；
- 数据处理使用 NumPy；
- 持久化使用 Python 标准库 SQLite 和 SQLite FTS5；
- 实际提交若包含来自其他论文、仓库或作者的代码，提交说明应按赛事要求补充原作者、来源链接和改动范围；
- 本服务不包含数据集专用答案表、Judge、查询 gold、evidence 驱动重排、提示词注入或结果操纵代码。

## 已知限制

- Docker 构建阶段需要访问 Hugging Face 下载模型；
- 模型会增加镜像大小和首次容器内加载时间；
- SQLite 适合当前评测规模，不适合百万级单用户记忆；
- 服务应保持单 worker；
