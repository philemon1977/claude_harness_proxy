# Claude Harness Proxy

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

**Claude 模型调用代理：自动修复工具参数 + 过滤特殊 Token + 多模态内容过滤，支持 Anthropic 和 OpenAI 双协议。**

[English README](README.md)

---

## 为什么存在这个项目

> "开源模型不擅长工具调用" — 几乎都是 **框架（harness）** 的问题，而非模型本身的能力问题。

作者仅通过几个微小的框架调整，就让 **DeepSeek V4 Pro** 在其内部工具调用评测中，**6/10 的次数击败了 Claude Opus 4.7** — 全程没有修改模型本身。

### 背景

作者花了两天时间分析 CommandCode（开源 AI CLI 工具）中 DeepSeek 产生的数十亿 token 日志，发现 **DeepSeek Flash** 连最简单的代码审查任务都会失败 — 每次调用 shell 命令或读取文件都会返回原始的 Zod 验证错误，而模型根本无法理解这种错误格式，自然无法自我修复。一层 **工具输入修复层** 彻底解决了这个问题。

### 关键发现

#### 失败模式高度固定且有限

包括 DeepSeek 全系列、GLM、Qwen 在内的主流开源模型，工具调用时几乎只会犯以下 **4 种重复错误**：

1. 给可选字段传 `null` 而不是直接省略
2. 把 JSON 数组写成字符串形式（`"[\"a\",\"b\"]"` 而非 `["a","b"]`）
3. 把单个参数用 `{}` 包裹，而 schema 期望的是数组
4. 本该传数组的地方传了纯字符串（`"foo"` 而非 `["foo"]`）

4 个修复函数，各 30–100 行，按正确顺序执行，解决了 **90% 的工具调用失败问题**。

#### 最隐蔽的 "训练泄露" 错误

DeepSeek Flash 有时会把文件路径写成 Markdown 自动链接：
`filePath: "[notes.md](notes.md)"` — 导致工具尝试创建字面意义上带括号的文件名。这不是幻觉，而是模型在聊天训练中被奖励自动链接，这种行为 "泄露" 到了工具调用场景。

**修复：** 仅用两行正则，专门处理链接文本等于 URL 的退化情况，不影响正常的 Markdown 链接。

#### 颠覆性的验证流程设计

最初的 "先预处理再验证" 思路会误改那些恰好是 JSON 格式的文件内容，造成静默损坏。

**改为 "先验证再修复"：**

1. 先原样解析输入，验证通过直接执行，绝不修改有效输入
2. 验证失败时，根据验证器返回的错误路径，按顺序尝试 4 种修复
3. 修复成功则记录日志，失败则返回模型能读懂的重试提示

这种方式让验证器自动定位错误位置，只在真正有问题的地方花费修复成本，还能免费获得每个模型、每个工具的错误率遥测数据。

#### 不同类型的错误需要不同的修复策略

上述 4 种修复都针对 **形状错误**（类型、结构不对），但还有 **关系错误**：比如 `readFile` 工具要求 `offset` 和 `limit` 必须成对出现，DeepSeek 经常只传其中一个。

**修复：** 扩展工具语义 — 只传 `limit` → 自动补全 `offset=0`；只传 `offset` → 自动补全 `limit=2000`（行业默认值）。返回结果中明确告知模型做了什么默认处理，让模型可以在下一轮自行纠正。

### 最终总结

很多看起来是 **"模型能力差距"** 的现象，本质上是 **契约设计的问题**。

商业大模型（如 Claude Opus）在预训练中见过海量不同格式的 API 契约，所以能自动容忍不严格的输入 — 这种成本被它们 "无形吃掉"。开源模型没有见过那么多契约，会在严格的 schema 下频繁失败，进而被误认为 "能力不行"。

**框架（harness）的核心作用，就是在模型的输出分布和工具的输入分布之间做调解。** 作者全程没有修改 DeepSeek 模型本身，只是让契约在恰好需要的地方变得更宽容，就实现了对 Claude Opus 4.7 的反超。

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
