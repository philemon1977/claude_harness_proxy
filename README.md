# Claude Harness Proxy

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

**Claude 模型调用代理：自动修复工具参数 + 过滤特殊 Token + 多模态内容过滤，支持 Anthropic 和 OpenAI 双协议。**

[Claude Harness Proxy](README_CN.md) — 中文 README

---

## Overview

Claude Harness Proxy sits between Claude Code CLI (or any LLM client) and upstream inference backends (vLLM, LiteLLM, llama.cpp). It intercepts API requests and responses, transparently applying:

1. **Tool Parameter Auto-Repair** — 12 repair rules fix malformed tool arguments caused by LLM output inconsistencies (null fields, type mismatches, trailing commas, typo parameter names, etc.).
2. **Gemma-4 Special Token Filter** — Removes leaked Gemma-4 control tokens (`<|tool_call|>`, `<|turn|>`, `<eos>`, etc.) from both streaming (SSE) and non-streaming responses.
3. **Multimodal Content Stripping** — Blocks image/video blocks and base64-encoded media from reaching local text-only backends, preventing `Unexpected item type` errors.

## Architecture

```
┌─────────────┐     POST /v1/messages      ┌──────────────────────────────────┐
│  Claude Code │ ──────────────────────────> │  Claude Harness Proxy           │
│  CLI (or any │                              │  :9200                         │
│  LLM client) │ <────────────────────────── │                                  │
│             │     filtered + repaired     │  ┌─ Multimodal Filter ──┐        │
└─────────────┘     JSON / SSE stream       │  ┌─ Tool Parameter Repair ──┐    │
                         │                  │  ┌─ Gemma-4 Token Filter ──┐    │
                         │ POST             │  └─────────────────────────┘    │
                         ▼               │  └─────────────────────────┘     │
┌──────────────────────────────────┐  │                                    │
│  Upstream (vLLM / LiteLLM)       │  │  Environment Variables:            │
│  :8200 / :4000                    │  │  UPSTREAM_URL  : http://127.0.0.1:8200│
│  (or llama.cpp / any HTTP API)  │  │  PROXY_HOST    : 127.0.0.1         │
│                                  │  │  PROXY_PORT    : 9200              │
└──────────────────────────────────┘  │  LOG_LEVEL     : INFO              │
                                     │  ENABLE_GEMMA4_FILTER: true       │
└───────────────────────────────────┘
```

## Features

### Tool Parameter Repair (12 Rules)

| # | Rule | Description |
|---|------|-----------|
| 1 | Strip null fields | Remove key-value pairs where value is `null` |
| 2 | Parse string-wrapped arrays | Convert `"[1,2,3]"` → `[1, 2, 3]` |
| 3 | Fix Markdown link paths | `[path](path)` → `path` |
| 4 | Auto-fill offset/limit | File read tools get default values |
| 5 | Fix string booleans | `"true"` / `"false"` → `True` / `False` |
| 6 | Fix string numbers | `"42"` → `42` (int/float by schema) |
| 7 | Fix trailing commas | `{a:1,}` → `{a:1}` |
| 8 | Remove extra quotes | `'value'` → `value` |
| 9 | Common param typos | `filepath` → `path`, `cmd` → `command` |
| 10 | Single-value wrapping | `"read"` → `["read"]` for array fields |
| 11 | Nested JSON strings | `'{"key":"val"}'` → `{"key":"val"}` |
| 12 | Pre-validation skip | Valid inputs pass through without processing |

### Gemma-4 Token Filtering

- **Non-streaming**: Cleans all content fields in JSON responses (both Anthropic and OpenAI formats)
- **Streaming (SSE)**: Line-buffered filter with fast-path passthrough — only parses JSON lines that contain special token markers
- Handles: `<|tool_call|>`, `<|turn|>`, `<|channel|>`, `<|think|>`, `<bos>`, `<eos>`, `<pad>`, `<unk>`, tool content blocks, thinking channel blocks

### Multimodal Content Filtering

- Strips `image_url` and `image` blocks from request content
- Detects base64-encoded images via magic number verification (PNG, JPEG, GIF, WebP, BMP, TIFF)
- Handles data URIs, bare base64 strings, and nested tool_result blocks
- Replaces removed content with `[image removed]` placeholder

## Quick Start

### Prerequisites

- Python 3.10+
- `pip install fastapi httpx uvicorn pydantic`

### Start the Proxy

```bash
# Default: listens on 127.0.0.1:7000, upstream http://127.0.0.1:4000
python claude_harness_proxy.py

# Custom configuration
UPSTREAM_URL=http://127.0.0.1:8200 \
PROXY_PORT=9200 \
LOG_LEVEL=DEBUG \
python claude_harness_proxy.py
```

### Configure Your Client

Point your Claude Code CLI or LLM client to the proxy:

```bash
# Instead of connecting directly to vLLM:8200
ANTHROPIC_API_BASE="http://127.0.0.1:9200/v1"
```

### Health Check

```bash
curl http://127.0.0.1:9200/health
# {"status": "ok", "upstream": "http://127.0.0.1:8200", ...}

curl http://127.0.0.1:9200/stats
# {"proxy": {"total_requests": ..., ...}, "filter": {...}}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Main proxy endpoint (Claude Code CLI) |
| `/{path:path}` | POST | Catch-all for other API paths |
| `/health` | GET | Health check + upstream status |
| `/stats` | GET | Repair/filter statistics |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSTREAM_URL` | `http://127.0.0.1:4000` | Upstream service address |
| `PROXY_HOST` | `127.0.0.1` | Listen address |
| `PROXY_PORT` | `7000` | Listen port |
| `LOG_LEVEL` | `INFO` | Log level (DEBUG/INFO/WARNING) |
| `ENABLE_GEMMA4_FILTER` | `true` | Enable/disable Gemma-4 token filtering |

## Project Structure

```
claude_harness_proxy/
├── claude_harness_proxy.py   # Main proxy server (FastAPI)
├── gemma4_token_filter.py    # Gemma-4 token filtering module
├── README.md                  # This file (English)
├── README_CN.md             # Chinese README
└── .gitignore
```

## Use Cases

- **Claude Code + Local vLLM**: Deploy between Claude Code CLI and a local vLLM backend to handle tool call parameter issues transparently
- **Gemma-4 Deployment**: Use Gemma-4 models without special tokens leaking into tool call output
- **Multimodal → Text-only**: Forward multimodal-capable client requests to text-only backends without errors
- **API Compatibility**: Bridge Anthropic Messages API format to OpenAI-compatible backends

## License

MIT License — see [LICENSE](LICENSE) for details.
