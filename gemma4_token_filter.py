"""
Gemma-4 Special Token Filter Module

Cleans Gemma-4 special control tokens from model output that leaked into
content text fields. Designed for both streaming (SSE) and non-streaming
response filtering in the Claude Harness Proxy.

Architecture:
  clean_content()        → non-streaming text cleanup
  clean_response_data()  → dual-protocol response cleanup (Anthropic + OpenAI)
  StreamingContentFilter → SSE streaming filter (line-buffered)
"""

import re
import json
import os


# ============================================================
# Configuration
# ============================================================
ENABLE_GEMMA4_FILTER = os.environ.get("ENABLE_GEMMA4_FILTER", "true").lower() in ("true", "1", "yes")

# ============================================================
# Compiled Regex Patterns (split by linearity per review feedback)
# ============================================================

# Token structure:
#   Self-closing: <|tool_call|> <|tool_response|> <|tool|> <|think|>
#   Opening:      <|channel> <|turn> <|tool_call> <|tool_response>
#   Closing:      <channel|> <turn|> <tool_call|> <tool_response|>
#   Standard:     <bos> <eos> <pad> <unk>
#   Delimiter:    <|"|>
# CRITICAL: Use [|] for literal pipe to avoid regex alternation

# Single-line tokens: run AFTER multi-line blocks (see clean_content)
# CRITICAL: Self-closing <|name|> alternatives must come BEFORE bare <|name>
# so that regex doesn't match the prefix and leave |> behind.
GEMMA4_SINGLE_LINE = re.compile(
    r'<\|turn\|>'
    r'|<\|turn>(?:model|user|system)\n?'
    r'|<\|turn>'
    r'|<turn\|>'
    r'|<\|think\|>'
    r'|<\|channel>'
    r'|<channel\|>'
    r'|<\|tool\|>'
    r'|<\|tool>'
    r'|<tool\|>'
    r'|<\|tool_call\|>'
    r'|<\|tool_call>'
    r'|<tool_call\|>'
    r'|<\|tool_response\|>'
    r'|<\|tool_response>'
    r'|<tool_response\|>'
    r'|<bos>|<eos>|<pad>|<unk>'
    r'|<\|"[|]\>'
)

# Multi-line thinking blocks: run BEFORE single-line tokens
# Handles <|channel>thought\n...\n<channel|> and the no-newline variant
GEMMA4_THINK_BLOCK = re.compile(
    r'<\|channel>thought\n.*?<channel\|>'
    r'|<\|channel>thought<channel\|>',
    re.DOTALL
)

# Tool content blocks: run BEFORE single-line tokens
# Matches content between opening and closing tool markers
GEMMA4_TOOL_BLOCKS = re.compile(
    r'<\|tool>.*?<tool\|>'
    r'|<\|tool_call>.*?<tool_call\|>'
    r'|<\|tool_response>.*?<tool_response\|>',
    re.DOTALL
)

# Whitespace collapse helper
_GEMMA4_COLLAPSE_WS = re.compile(r' +')


# ============================================================
# Statistics (for /stats endpoint)
# ============================================================
filter_stats = {
    "single_line_hits": 0,
    "tool_token_hits": 0,
    "think_block_hits": 0,
    "stream_filter_attempts": 0,
    "stream_filter_skipped": 0,
}


# ============================================================
# Non-streaming cleanup
# ============================================================

def clean_content(text: str) -> str:
    """
    Remove all Gemma-4 special tokens from content text.

    Order: think blocks → tool blocks → single-line tokens → whitespace cleanup.
    Multi-line blocks run first so their markers are consumed before
    GEMMA4_SINGLE_LINE processes remaining standalone tokens.
    """
    if not text:
        return text

    orig = text

    # Step 1: Thinking channel blocks (multi-line, run first)
    after_think = GEMMA4_THINK_BLOCK.sub('', text)
    if after_think != text:
        filter_stats["think_block_hits"] += 1
        text = after_think

    # Step 2: Tool content blocks (multi-line, run before single-line)
    after_tool = GEMMA4_TOOL_BLOCKS.sub('', text)
    if after_tool != text:
        filter_stats["tool_token_hits"] += 1
        text = after_tool

    # Step 3: Single-line tokens
    text = GEMMA4_SINGLE_LINE.sub('', text)
    if text != orig and text == after_tool:
        filter_stats["single_line_hits"] += 1

    # Step 4: Collapse whitespace artifacts
    text = _GEMMA4_COLLAPSE_WS.sub(' ', text)

    # Step 5: Trim
    return text.strip()


def clean_response_data(resp_data: dict) -> dict:
    """
    Clean special tokens from all content fields in a response.
    Handles both Anthropic and OpenAI response formats.
    """
    if not ENABLE_GEMMA4_FILTER or not isinstance(resp_data, dict):
        return resp_data

    # Anthropic format: content is a list of blocks
    if "content" in resp_data and isinstance(resp_data["content"], list):
        for block in resp_data["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                block["text"] = clean_content(block.get("text", ""))

    # OpenAI format: content is a string in choices[0].message
    if "choices" in resp_data and isinstance(resp_data["choices"], list):
        for choice in resp_data["choices"]:
            if isinstance(choice, dict):
                msg = choice.get("message", {})
                if isinstance(msg, dict) and "content" in msg and msg["content"]:
                    msg["content"] = clean_content(str(msg["content"]))

    return resp_data


# ============================================================
# Streaming SSE Filter
# ============================================================

# (Removed — fast check is done with simple string containment)


class StreamingContentFilter:
    """
    Filters Gemma-4 special tokens from SSE streams.

    Design:
    - Operates line-by-line on SSE data (split on \n)
    - Fast path: line has no special token markers → passthrough
    - Slow path: parse JSON → clean text fields → re-serialize
    - Incomplete lines buffered for next chunk
    """

    def __init__(self):
        self._buf = ""

    def feed(self, chunk: bytes) -> bytes:
        """Process incoming bytes and return filtered output."""
        try:
            chunk_str = chunk.decode("utf-8", errors="replace")
        except Exception:
            return chunk

        self._buf += chunk_str
        return self._drain().encode("utf-8")

    def drain(self) -> bytes:
        """Drain remaining buffer."""
        if not self._buf:
            return b""
        remaining = self._drain()
        self._buf = ""
        return remaining.encode("utf-8") if remaining else b""

    def reset(self):
        """Reset for a new stream."""
        self._buf = ""

    def _drain(self) -> str:
        output = []
        while True:
            idx = self._buf.find("\n")
            if idx == -1:
                break

            line = self._buf[:idx]
            self._buf = self._buf[idx + 1:]

            if not ENABLE_GEMMA4_FILTER:
                output.append(line + "\n")
                continue

            if line.startswith("data:"):
                output.append(self._filter_data_line(line))
            else:
                output.append(line + "\n")

        return "".join(output)

    def _filter_data_line(self, line: str) -> str:
        """Filter a single SSE data line."""
        filter_stats["stream_filter_attempts"] += 1

        # Fast path: no special token markers → passthrough
        payload = line[5:]  # skip "data:"
        if "<|" not in payload and "<bos>" not in payload and "<eos>" not in payload and "<pad>" not in payload and "|>" not in payload and "<tool|" not in payload and "<channel|" not in payload and "<turn|" not in payload:
            filter_stats["stream_filter_skipped"] += 1
            return line + "\n"

        # Slow path: parse JSON, clean text fields, re-serialize
        try:
            payload_stripped = payload.strip()
            if payload_stripped.startswith(" "):
                payload_stripped = payload_stripped[1:]
            data = json.loads(payload_stripped)
            self._clean_json_obj(data)
            return "data: " + json.dumps(data, ensure_ascii=False) + "\n"
        except (json.JSONDecodeError, ValueError, TypeError):
            # Partial JSON at chunk boundary — pass through
            return line + "\n"

    def _clean_json_obj(self, obj):
        """Recursively clean text fields in JSON."""
        if isinstance(obj, str):
            pass  # We only clean specific keys, not all strings
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("text", "content", "thinking") and isinstance(v, str):
                    obj[k] = clean_content(v)
                elif isinstance(v, (dict, list)):
                    self._clean_json_obj(v)
        elif isinstance(obj, list):
            for item in obj:
                self._clean_json_obj(item)
