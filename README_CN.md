# Claude Harness Proxy

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

**Claude 模型调用代理：自动修复工具参数 + 过滤特殊 Token + 多模态内容过滤，支持 Anthropic 和 OpenAI 双协议。**

[English README](README.md)

---

## 项目概述

Claude Harness Proxy 部署在 Claude Code CLI（或任意 LLM 客户端）与上游推理后端（vLLM、LiteLLM、llama.cpp）之间。它拦截 API 请求和响应，透明地执行以下处理：

1. **工具参数自动修复** — 12 项修复规则，自动修正 LLM 输出导致的工具参数错误（null 字段、类型不匹配、尾部逗号、参数名拼写错误等）。
2. **Gemma-4 特殊 Token 过滤** — 从流式（SSE）和非流式响应中清除泄漏的 Gemma-4 控制 token（`<|tool_call|>`、`<|turn|>`、`<eos>` 等）。
3. **多模态内容过滤** — 阻止图片/视频块和 base64 编码媒体传递给本地纯文本后端，避免 `Unexpected item type` 错误。

## 架构

```
┌─────────────┐     POST /v1/messages      ┌──────────────────────────────────┐
│  Claude Code │ ──────────────────────────> │  Claude Harness Proxy           │
│  CLI（或任  │                              │  :9200                         │
│  意 LLM 客户端）│ <────────────────────────── │                                  │
│             │     过滤+修复后的             │  ┌─ 多模态内容过滤 ──┐        │
└─────────────┘     JSON / SSE 流          │  ┌─ 工具参数修复 ──────────┐    │
                         │                  │  ┌─ Gemma-4 Token 过滤 ──┐    │
                         │ POST             │  └─────────────────────────┘    │
                         ▼               │  └─────────────────────────┘     │
┌──────────────────────────────────┐  │                                    │
│  上游服务（vLLM / LiteLLM）       │  │  环境变量配置：                     │
│  :8200 / :4000                    │  │  UPSTREAM_URL  : http://127.0.0.1:8200│
│  或 llama.cpp / 任意 HTTP API    │  │  PROXY_HOST    : 127.0.0.1         │
│                                  │  │  PROXY_PORT    : 9200              │
└──────────────────────────────────┘  │  LOG_LEVEL     : INFO              │
                                     │  ENABLE_GEMMA4_FILTER: true        │
└───────────────────────────────────┘
```

## 功能特性

### 工具参数修复（12 项规则）

| # | 规则 | 说明 |
|---|------|------|
| 1 | 删除 null 字段 | 移除值为 `null` 的键值对 |
| 2 | 解析字符串包裹的数组 | `"[1,2,3]"` → `[1, 2, 3]` |
| 3 | 修复 Markdown 链接路径 | `[path](path)` → `path` |
| 4 | 自动补全 offset/limit | 文件读取工具自动填充默认值 |
| 5 | 修复字符串布尔值 | `"true"` / `"false"` → `True` / `False` |
| 6 | 修复字符串数字 | `"42"` → `42`（根据 schema 转为 int/float） |
| 7 | 修复尾部逗号 | `{a:1,}` → `{a:1}` |
| 8 | 移除多余引号 | `'value'` → `value` |
| 9 | 常见参数名纠错 | `filepath` → `path`，`cmd` → `command` |
| 10 | 单值包装为数组 | `"read"` → `["read"]`（针对数组字段） |
| 11 | 嵌套 JSON 反序列化 | `'{"key":"val"}'` → `{"key":"val"}` |
| 12 | 预校验快速通道 | 合法输入直接跳过处理 |

### Gemma-4 Token 过滤

- **非流式**：清理 JSON 响应中所有 content 字段（支持 Anthropic 和 OpenAI 格式）
- **流式（SSE）**：行缓冲过滤器，快速路径直接透传 — 仅解析包含特殊 token 标记的 JSON 行
- 覆盖：`<|tool_call|>`、`<|turn|>`、`<|channel|>`、`<|think|>`、`<bos>`、`<eos>`、`<pad>`、`<unk>`、tool 内容块、thinking 通道块

### 多模态内容过滤

- 从请求内容中移除 `image_url` 和 `image` block
- 通过魔数验证检测 base64 编码图片（PNG, JPEG, GIF, WebP, BMP, TIFF）
- 处理 data URI、裸 base64 字符串、嵌套 tool_result 块
- 移除的内容替换为 `[image removed]` 占位符

## 快速开始

### 环境要求

- Python 3.10+
- `pip install fastapi httpx uvicorn pydantic`

### 启动代理

```bash
# 默认：监听 127.0.0.1:7000，上游 http://127.0.0.1:4000
python claude_harness_proxy.py

# 自定义配置
UPSTREAM_URL=http://127.0.0.1:8200 \
PROXY_PORT=9200 \
LOG_LEVEL=DEBUG \
python claude_harness_proxy.py
```

### 配置客户端

将 Claude Code CLI 或 LLM 客户端指向代理：

```bash
# 不再直连 vLLM:8200，而是通过代理
ANTHROPIC_API_BASE="http://127.0.0.1:9200/v1"
```

### 健康检查

```bash
curl http://127.0.0.1:9200/health
# {"status": "ok", "upstream": "http://127.0.0.1:8200", ...}

curl http://127.0.0.1:9200/stats
# {"proxy": {"total_requests": ..., ...}, "filter": {...}}
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/messages` | POST | 主代理端点（Claude Code CLI） |
| `/{path:path}` | POST | 兼容其他 API 路径 |
| `/health` | GET | 健康检查 + 上游状态 |
| `/stats` | GET | 修复/过滤统计信息 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `UPSTREAM_URL` | `http://127.0.0.1:4000` | 上游服务地址 |
| `PROXY_HOST` | `127.0.0.1` | 监听地址 |
| `PROXY_PORT` | `7000` | 监听端口 |
| `LOG_LEVEL` | `INFO` | 日志级别（DEBUG/INFO/WARNING） |
| `ENABLE_GEMMA4_FILTER` | `true` | 启用/禁用 Gemma-4 token 过滤 |

## 项目结构

```
claude_harness_proxy/
├── claude_harness_proxy.py   # 主代理服务（FastAPI）
├── gemma4_token_filter.py    # Gemma-4 token 过滤模块
├── README.md                  # 中文 README
├── README_CN.md             # 英文 README
└── .gitignore
```

## 使用场景

- **Claude Code + 本地 vLLM**：在 Claude Code CLI 和本地 vLLM 后端之间部署，透明处理工具调用参数问题
- **Gemma-4 部署**：使用 Gemma-4 模型时，防止特殊 token 泄漏到工具调用输出中
- **多模态 → 纯文本**：将支持多模态的客户端请求转发到纯文本后端，不会产生错误
- **API 兼容**：桥接 Anthropic Messages API 格式到 OpenAI 兼容后端

## 许可证

MIT License — 详见 [LICENSE](LICENSE)。
