#!/usr/bin/env python3
"""
Claude Harness Proxy - 工具调用参数自动修复 + Gemma-4 特殊 Token 过滤代理
专为 Claude Code CLI (Anthropic Messages API) 设计，支持双协议（Anthropic + OpenAI）。

功能：
1. 工具参数自动修复（12 项规则）
2. Gemma-4 特殊 token 过滤（流式 + 非流式）

接收 Anthropic 请求 → 修复 tool_use 参数 → 过滤特殊 token → 转发到上游 (LiteLLM / vLLM / llama.cpp)
"""

import re
import json
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager
from typing import Any, Dict, Type, Union, Optional
from functools import lru_cache
from collections import defaultdict
import types
from typing import get_origin, get_args

sys.path.insert(0, os.path.dirname(__file__))
from gemma4_token_filter import clean_response_data, StreamingContentFilter, filter_stats, ENABLE_GEMMA4_FILTER  # noqa: E402

import base64
import binascii

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from pydantic import BaseModel, create_model, ValidationError

# ==================================================
# 配置（通过环境变量覆盖）
# ==================================================
UPSTREAM_URL = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:4000")   # 上游服务地址 (LiteLLM / vLLM / llama.cpp)
PROXY_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "7000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# 日志配置（按端口区分文件，支持多实例并行）
LOG_DIR = os.path.expanduser("~/.claude-harness")
LOG_FILE = os.path.join(LOG_DIR, f"harness-{PROXY_PORT}.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 允许透传的请求头白名单
ALLOWED_HEADERS = {
    "authorization", "anthropic-version", "content-type",
    "anthropic-beta", "x-api-key"
}

# 统计信息
stats = defaultdict(int)

# ==================================================
# 一、工具参数修复引擎（12项规则）
# ==================================================

def strip_null_fields(obj: Dict, recursive: bool = True) -> Dict:
    """Fix1: 删除值为null的字段"""
    result = {}
    for k, v in obj.items():
        if v is None:
            continue
        if recursive and isinstance(v, dict):
            result[k] = strip_null_fields(v, recursive=True)
        else:
            result[k] = v
    return result

def parse_string_wrapped_array(data: Any) -> Any:
    """Fix2: 字符串包裹的JSON数组转原生数组"""
    if isinstance(data, str):
        stripped = data.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return data

def fix_md_file_path(path: str) -> str:
    """Fix3: 修复Markdown链接格式畸形路径"""
    pattern = r"\[(.+?)\]\(\1\)"
    return re.sub(pattern, r"\1", path)

def fix_offset_limit(params: Dict) -> Dict:
    """Fix4: 文件读取工具缺参自动补全"""
    res = params.copy()
    if "limit" in res and "offset" not in res:
        res["offset"] = 0
    if "offset" in res and "limit" not in res:
        res["limit"] = 2000
    return res

def fix_string_booleans(obj: Dict) -> Dict:
    """Fix5: 字符串格式布尔值转原生类型"""
    res = obj.copy()
    for k, v in res.items():
        if isinstance(v, str):
            lower_v = v.lower()
            if lower_v == "true":
                res[k] = True
            elif lower_v == "false":
                res[k] = False
    return res

def fix_string_numbers(obj: Dict, schema_cls: Type[BaseModel]) -> Dict:
    """
    Fix6: 字符串格式数字转原生数值
    支持 int | None, float | None, 原生 int/float
    """
    res = obj.copy()
    fields = schema_cls.model_fields
    for k, v in res.items():
        if k not in fields or not isinstance(v, str):
            continue
        field_type = fields[k].annotation
        origin = get_origin(field_type)
        # 处理 Union 类型（包括 int|None）
        if origin in (Union, types.UnionType):
            args = get_args(field_type)
            try:
                if int in args:
                    res[k] = int(v)
                elif float in args:
                    res[k] = float(v)
            except (ValueError, TypeError):
                continue
        elif field_type is int:
            try:
                res[k] = int(v)
            except ValueError:
                pass
        elif field_type is float:
            try:
                res[k] = float(v)
            except ValueError:
                pass
    return res

def fix_trailing_comma(json_str: str) -> str:
    """Fix7: 修复JSON尾部多余逗号"""
    return re.sub(r",\s*([}\]])", r"\1", json_str)

def fix_extra_quotes(obj: Dict) -> Dict:
    """Fix8: 修复字段值首尾多余引号"""
    res = obj.copy()
    for k, v in res.items():
        if isinstance(v, str) and len(v) >= 2:
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                res[k] = v[1:-1]
    return res

def fix_common_param_typos(params: Dict) -> Dict:
    """Fix9: 常见参数名拼写纠错"""
    typo_map = {
        "filepath": "path",
        "file_path": "path",
        "filename": "path",
        "file_name": "path",
        "cmd": "command",
        "command_line": "command",
        "dir": "cwd",
        "directory": "cwd",
        "offest": "offset",
        "limt": "limit",
    }
    res = params.copy()
    for wrong, right in typo_map.items():
        if wrong in res and right not in res:
            res[right] = res.pop(wrong)
    return res

def fix_single_value_wrapped(obj: Dict, schema_cls: Type[BaseModel]) -> Dict:
    """Fix10: 单值自动包装为数组"""
    res = obj.copy()
    fields = schema_cls.model_fields
    for k, v in res.items():
        if k not in fields:
            continue
        field_type = fields[k].annotation
        origin = get_origin(field_type)
        is_array = (origin is list) or (field_type is list)
        if is_array and not isinstance(v, list):
            res[k] = [v]
    return res

def fix_nested_string_json(obj: Dict) -> Dict:
    """Fix11: 嵌套对象字符串反序列化"""
    res = obj.copy()
    for k, v in res.items():
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    res[k] = json.loads(stripped)
                except json.JSONDecodeError:
                    pass
    return res

# ==================================================
# 修复调度器（优先级：清理 → 类型转换 → 结构适配）
# ==================================================
def repair_tool_input(raw_input: Dict, schema_cls: Type[BaseModel]) -> Dict:
    """执行分层修复，返回修复后的参数"""
    # 原样校验，合法直接返回
    try:
        schema_cls.model_validate(raw_input)
        return raw_input
    except ValidationError as e:
        logger.debug(f"原始参数校验失败: {e.errors()}")

    fixed = raw_input.copy()
    stats["repair_attempts"] += 1

    # 按优先级执行修复
    fixed = strip_null_fields(fixed, recursive=True)
    fixed = fix_extra_quotes(fixed)
    fixed = fix_common_param_typos(fixed)
    fixed = fix_string_booleans(fixed)
    fixed = fix_string_numbers(fixed, schema_cls)

    for k, v in fixed.items():
        fixed[k] = parse_string_wrapped_array(v)

    fixed = fix_single_value_wrapped(fixed, schema_cls)
    fixed = fix_nested_string_json(fixed)

    if "path" in fixed and isinstance(fixed["path"], str):
        fixed["path"] = fix_md_file_path(fixed["path"])

    fixed = fix_offset_limit(fixed)

    # 二次校验
    try:
        schema_cls.model_validate(fixed)
        stats["repair_success"] += 1
        logger.info("已自动修复工具参数错误")
        return fixed
    except ValidationError as e:
        stats["repair_failed"] += 1
        logger.debug(f"修复后仍校验失败: {e.errors()}")
        return raw_input

# ==================================================
# 二、动态 Schema 生成（支持 Anthropic 与 OpenAI 格式）
# ==================================================
def _map_json_type(field_schema: Dict) -> Type:
    type_ = field_schema.get("type")
    if type_ == "string":
        return str
    elif type_ == "integer":
        return int
    elif type_ == "number":
        return float
    elif type_ == "boolean":
        return bool
    elif type_ == "array":
        return list
    elif type_ == "object":
        return dict
    else:
        return str

def json_schema_to_pydantic(schema: Dict, tool_name: str) -> Type[BaseModel]:
    """
    支持两种格式的 Schema 转换：
    1. Anthropic: { "name": "...", "input_schema": { "properties": {...}, "required": [...] } }
    2. OpenAI: { "name": "...", "parameters": { "type": "object", "properties": {...}, "required": [...] } }
    """
    # 自动识别是 Anthropic 还是 OpenAI 格式
    if "input_schema" in schema:
        spec = schema.get("input_schema", {})
    elif "parameters" in schema:
        spec = schema.get("parameters", {})
    else:
        spec = {}

    properties = spec.get("properties", {})
    required_fields = spec.get("required", [])
    fields_def = {}

    for field_name, field_schema in properties.items():
        python_type = _map_json_type(field_schema)
        if field_name not in required_fields:
            fields_def[field_name] = (python_type | None, None)
        else:
            fields_def[field_name] = (python_type, ...)

    return create_model(f"{tool_name}_schema", **fields_def)

@lru_cache(maxsize=128)
def get_tool_schema(tool_name: str, schema_json: str) -> Type[BaseModel]:
    """缓存工具对应的 Pydantic 模型"""
    schema = json.loads(schema_json)
    return json_schema_to_pydantic(schema, tool_name)

# ==================================================
# 三、多模态内容过滤（严禁向本地推理引擎发送图片）
# ==================================================

def _is_likely_base64_image(text: str, threshold: int = 64) -> bool:
    """
    检测字符串是否为内嵌的 base64 编码图片。

    支持三种场景：
    1. 裸 base64 字符串
    2. data URI (data:image/png;base64,...)
    3. 文本中嵌入 data URI 或 base64 (如 "see: data:image/...")

    解码后前缀匹配常见图片格式魔数（PNG/JPEG/GIF/WebP/BMP/TIFF）。
    """
    text_lower = text.lower().strip()

    # 场景2/3: data URI（可能带前缀文本）
    if "base64," in text_lower:
        comma_idx = text_lower.index("base64,")
        b64_part = text[comma_idx + 7:].strip()
        # 如果 data: 前缀中包含 image/ 或 video/
        prefix_part = text[:comma_idx + 7]
        if "image/" in prefix_part.lower() or "video/" in prefix_part.lower():
            try:
                decoded = base64.b64decode(b64_part, validate=True)
                return len(decoded) > 0
            except (binascii.Error, ValueError):
                pass

    stripped = text.strip()
    if len(stripped) < threshold:
        return False

    # 场景1: 纯 data URI (无前缀文本)
    if stripped.lower().startswith("data:") and "base64," in stripped:
        comma_idx = stripped.index("base64,")
        stripped = stripped[comma_idx + 7:]

    try:
        decoded = base64.b64decode(stripped, validate=True)
    except (binascii.Error, ValueError):
        return False
    magic = {
        b'\x89PNG\r\n': 'png',
        b'\xff\xd8\xff': 'jpeg',
        b'GIF8': 'gif',
        b'\x00\x00\x00\x0cftype': 'ftyp',
        b'BM': 'bmp',
        b'II*\x00': 'tiff',
        b'MM\x00*': 'tiff',
    }
    for prefix, _ in magic.items():
        if decoded.startswith(prefix):
            return True
    return False


def strip_multimodal_anthropic_content(content_blocks: list) -> list:
    """
    从 Anthropic content block 列表中移除 image/video/tool_reference block。

    Anthropic block 格式：
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
    {"type": "video", "source": {...}}
    {"type": "tool_reference", ...}
    {"type": "tool_result", "content": [...]}  ← 递归清洗
    """
    result = []
    for block in content_blocks:
        if not isinstance(block, dict):
            result.append(block)
            continue

        block_type = block.get("type")

        if block_type in ("image", "video", "tool_reference"):
            stats["stripped_image_blocks"] += 1
            continue

        if block_type == "tool_result" and "content" in block and isinstance(block["content"], list):
            block = dict(block)
            block["content"] = strip_multimodal_anthropic_content(block["content"])

        if block_type == "text" and "text" in block:
            text = block["text"]
            if _is_likely_base64_image(text):
                stats["stripped_text_base64"] += 1
                block = dict(block)
                block["text"] = "[image removed]"

        result.append(block)
    return result


def strip_multimodal_openai_content(content: Any) -> Any:
    """
    从 OpenAI content 中移除 image_url block。

    OpenAI content 格式：
    - string: 纯文本（可能含 base64）
    - array: [{"type":"text","text":"..."}, {"type":"image_url","image_url":{"url":"..."}}]
    """
    if isinstance(content, str):
        if _is_likely_base64_image(content):
            stats["stripped_text_base64"] += 1
            return "[image removed]"
        return content

    if isinstance(content, list):
        result = []
        for block in content:
            if not isinstance(block, dict):
                result.append(block)
                continue
            if block.get("type") == "image_url":
                stats["stripped_image_blocks"] += 1
                continue
            if block.get("type") == "text" and "text" in block:
                text = block["text"]
                if _is_likely_base64_image(text):
                    stats["stripped_text_base64"] += 1
                    result.append({**block, "text": "[image removed]"})
                else:
                    result.append(block)
            else:
                result.append(block)
        return result

    return content


def strip_multimodal_request(body: dict) -> dict:
    """
    从请求体中剥离所有多模态内容（图片/视频）。

    处理 Anthropic 和 OpenAI 两种消息格式，覆盖 messages[]、system 字段。
    """
    messages = body.get("messages", [])
    if not messages:
        return body

    result_msgs = []
    for msg in messages:
        if not isinstance(msg, dict):
            result_msgs.append(msg)
            continue

        new_msg = dict(msg)
        content = msg.get("content")

        if isinstance(content, list):
            if content and any(isinstance(b, dict) and b.get("type") == "image_url" for b in content):
                new_msg["content"] = strip_multimodal_openai_content(content)
            else:
                new_msg["content"] = strip_multimodal_anthropic_content(content)
        elif isinstance(content, str):
            if _is_likely_base64_image(content):
                new_msg["content"] = "[image removed]"

        # system role 额外清洗
        if msg.get("role") == "system":
            sys_content = new_msg.get("content")
            if isinstance(sys_content, list):
                new_msg["content"] = strip_multimodal_anthropic_content(sys_content)
            elif isinstance(sys_content, str) and _is_likely_base64_image(sys_content):
                new_msg["content"] = "[image removed]"

        result_msgs.append(new_msg)

    body["messages"] = result_msgs

    # 清洗顶层 system 字段（Anthropic 格式）
    if "system" in body:
        sys_content = body["system"]
        if isinstance(sys_content, list):
            body["system"] = strip_multimodal_anthropic_content(sys_content)
        elif isinstance(sys_content, str) and _is_likely_base64_image(sys_content):
            body["system"] = "[image removed]"

    return body


# ==================================================
# 四、代理服务（Anthropic Messages API 适配）
# ==================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient(timeout=300.0) as client:
        app.state.client = client
        logger.info(f"Claude Harness 代理启动")
        logger.info(f"监听地址: {PROXY_HOST}:{PROXY_PORT}")
        logger.info(f"上游地址: {UPSTREAM_URL}")
        logger.info(f"日志文件: {LOG_FILE}")
        logger.info(f"Gemma-4 token filter: {'enabled' if ENABLE_GEMMA4_FILTER else 'disabled'}")
        logger.info("流式请求进行 token 过滤 + 透传，非流式请求进行工具参数修复 + token 过滤")
        logger.info("多模态内容过滤：已启用（图片/视频 → 文本请求）")
        yield
    logger.info("代理服务已停止")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok", "upstream": UPSTREAM_URL, "stats": dict(stats)}

@app.get("/stats")
async def stats_endpoint():
    return {"proxy": dict(stats), "filter": filter_stats.copy()}

@app.post("/v1/messages")  # Claude Code CLI 调用的端点
@app.post("/{path:path}")   # 兼容其他路径
async def proxy(request: Request, path: str = "v1/messages"):
    # 读取请求体
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"无效请求体: {str(e)}")
        raise HTTPException(status_code=400, detail="无效请求体")

    stats["total_requests"] += 1

    # 剥离多模态内容（严禁向本地推理引擎发送图片/视频/tool_reference）
    body = strip_multimodal_request(body)

    # 提取工具定义，生成校验模型（用于修复响应中的 tool_use / tool_calls）
    tool_schemas = {}
    # 1. 适配 Anthropic 格式 ("tools" 列表)
    if "tools" in body:
        for tool in body["tools"]:
            tool_name = tool.get("name")
            if tool_name:
                schema_json = json.dumps(tool, sort_keys=True)
                try:
                    tool_schemas[tool_name] = get_tool_schema(tool_name, schema_json)
                except Exception as e:
                    logger.warning(f"生成工具 {tool_name} (Anthropic) 的 Schema 失败: {e}")

    # 2. 适配 OpenAI 格式 ("tools" 列表, 结构略有不同)
    elif "tools" not in body and "messages" in body:
        # LiteLLM OpenAI 模式下 tools 可能在请求体根目录或通过其他方式传递
        # 这里做通用处理，如果 body 中有 tools 则尝试解析
        pass

    # 补充：如果 LiteLLM 转发的是 OpenAI 格式，tools 字段依然存在但结构为 {"type": "function", "function": {...}}
    if "tools" in body:
        for tool in body["tools"]:
            if "function" in tool:
                func_spec = tool["function"]
                tool_name = func_spec.get("name")
                if tool_name:
                    # 将 OpenAI function 格式转换为 get_tool_schema 可识别的格式
                    wrapped_schema = {"name": tool_name, "parameters": func_spec}
                    schema_json = json.dumps(wrapped_schema, sort_keys=True)
                    try:
                        tool_schemas[tool_name] = get_tool_schema(tool_name, schema_json)
                    except Exception as e:
                        logger.warning(f"生成工具 {tool_name} (OpenAI) 的 Schema 失败: {e}")

    # 构建请求头（白名单过滤）
    headers = {}
    for key, value in request.headers.items():
        if key.lower() in ALLOWED_HEADERS:
            headers[key] = value
    headers.pop("host", None)
    headers.pop("content-length", None)

    target_url = f"{UPSTREAM_URL}/{path.lstrip('/')}"

    # 流式请求：token 过滤 + 透传
    if body.get("stream", False):
        stats["stream_requests"] += 1
        try:
            req = app.state.client.build_request("POST", target_url, json=body, headers=headers)
            upstream_resp = await app.state.client.send(req, stream=True)
        except Exception as e:
            logger.error(f"上游请求失败: {str(e)}")
            stats["upstream_errors"] += 1
            raise HTTPException(status_code=502, detail=f"上游请求失败: {str(e)}")

        async def forward_stream():
            sfilter = StreamingContentFilter()
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    filtered = sfilter.feed(chunk)
                    if filtered:
                        yield filtered
            finally:
                remaining = sfilter.drain()
                if remaining:
                    yield remaining
        return StreamingResponse(
            forward_stream(),
            media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
            status_code=upstream_resp.status_code
        )

    # 非流式请求：转发并修复响应中的 tool_use 参数
    try:
        response = await app.state.client.post(target_url, json=body, headers=headers)
    except Exception as e:
        logger.error(f"转发失败: {str(e)}")
        stats["upstream_errors"] += 1
        raise HTTPException(status_code=502, detail=f"转发失败: {str(e)}")

    # 上游错误直接返回
    if response.status_code >= 400:
        return JSONResponse(
            content=response.text,
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items()
                     if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")}
        )

    # 解析 JSON 响应
    try:
        resp_data = response.json()
    except Exception:
        return JSONResponse(content=response.text, status_code=response.status_code)

    # Gemma-4 特殊 token 过滤（在 tool parameter repair 之前执行）
    resp_data = clean_response_data(resp_data)

    # ------------------------------------------------------------------
    # 修复响应中的工具参数（双协议适配）
    # ------------------------------------------------------------------

    # 模式 A: Anthropic 格式 (content 列表中包含 tool_use)
    if "content" in resp_data and isinstance(resp_data["content"], list):
        for i, block in enumerate(resp_data["content"]):
            if block.get("type") == "tool_use":
                tool_name = block.get("name")
                raw_input = block.get("input", {})
                if tool_name in tool_schemas:
                    fixed_input = repair_tool_input(raw_input, tool_schemas[tool_name])
                    resp_data["content"][i]["input"] = fixed_input

    # 模式 B: OpenAI 格式 (choices[0].message.tool_calls)
    elif "choices" in resp_data and isinstance(resp_data["choices"], list):
        for choice in resp_data["choices"]:
            message = choice.get("message", {})
            if "tool_calls" in message and isinstance(message["tool_calls"], list):
                for tool_call in message["tool_calls"]:
                    func = tool_call.get("function", {})
                    tool_name = func.get("name")
                    raw_args_str = func.get("arguments", "{}")

                    if tool_name in tool_schemas:
                        try:
                            # OpenAI 参数是 JSON 字符串，需要先反序列化
                            raw_input = json.loads(raw_args_str)
                            fixed_input = repair_tool_input(raw_input, tool_schemas[tool_name])
                            # 修复后再序列化回字符串
                            func["arguments"] = json.dumps(fixed_input, ensure_ascii=False)
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.debug(f"OpenAI 参数反序列化失败: {e}")

    # 返回修复后的响应
    return JSONResponse(
        content=resp_data,
        status_code=response.status_code,
        headers={k: v for k, v in response.headers.items()
                 if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")}
    )

# ==================================================
# 五、启动入口
# ==================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=PROXY_HOST,
        port=PROXY_PORT,
        log_level=LOG_LEVEL.lower(),
        access_log=False
    )
