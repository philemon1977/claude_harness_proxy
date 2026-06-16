#!/bin/bash
# Example: Gemma-4-31B NVFP4 + Harness Proxy
# Description: vLLM -> Harness Proxy chain
# Modify the variables below for your environment before running.
set -euo pipefail

# ── 基础路径与环境 ────────────────────────────────────────────────────────────────
# 修改为你的 vLLM Python 环境路径
VLLM_PYTHON="${VLLM_PYTHON:-/home/qiba/miniconda/envs/vllm022_py311/bin/python}"
HARNESS_PYTHON="${VLLM_PYTHON}"

# 修改为你的 GPU 配置
GPU_DEVICE="${GPU_DEVICE:-0}"

# ── 端口定义 ──────────────────────────────────────────────────────────────────
VLLM_PORT="${VLLM_PORT:-8100}"
HARNESS_PORT="${HARNESS_PORT:-9100}"

# ── 环境变量注入 ──────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_SLEEP_WHEN_IDLE="1"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export VLLM_TORCH_COMPILE_CACHE_DIR="${HOME}/.cache/vllm/torch_compile_cache"

# ── 模型配置 ──────────────────────────────────────────────────────────────────
# 修改为你的模型路径
MODEL_PATH="${MODEL_PATH:-/home/qiba/ai/models/Gemma-4-31B-IT-NVFP4}"
ASSISTANT_PATH="${MODEL_PATH}/assistant"
LOG_FILE="/tmp/vllm-harness-gemma4-${VLLM_PORT}.log"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
MAX_MODEL_LEN=262144
MAX_NUM_SEQS=4
MAX_BATCHED_TOKENS=32768
BLOCK_SIZE=64
SPEC_TOKENS=4

# 1. 预编译 FlashInfer 内核 (vLLM 0.6.x+ 必要)
echo "Step 0: Precompiling FlashInfer kernels..."
mkdir -p "${HOME}/.cache/flashinfer" "${VLLM_TORCH_COMPILE_CACHE_DIR}"
"${VLLM_PYTHON}" -c "
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '${GPU_DEVICE}'
import torch
from flashinfer.attention import BatchAttention
_ = BatchAttention()
from flashinfer import norm, rope
print('FlashInfer precompile done')
" 2>&1 || echo "FlashInfer precompile warning (non-fatal)"

# 2. 启动 vLLM 引擎
echo "Step 1: Starting vLLM Engine on port ${VLLM_PORT}..."
> "${LOG_FILE}"
nohup "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --quantization compressed-tensors \
  --served-model-name gemma4-31b \
  --tool-call-parser gemma4 \
  --host 0.0.0.0 --port "${VLLM_PORT}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
  --block-size "${BLOCK_SIZE}" \
  --speculative-config "{\"method\":\"mtp\", \"model\":\"${ASSISTANT_PATH}\",\"num_speculative_tokens\":${SPEC_TOKENS}}" \
  --dtype auto \
  --tokenizer-mode auto \
  --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
  --no-use-tqdm-on-load \
  --default-chat-template-kwargs '{"preserve_thinking": false, "enable_thinking": false}' \
  >> "${LOG_FILE}" 2>&1 &

# 等待 vLLM 就绪
until curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null; do
  echo "Waiting for vLLM to initialize..."
  sleep 5
done
echo "vLLM Engine is UP!"

# 3. 启动 Claude Harness Proxy
echo "Step 2: Starting Claude Harness Proxy on port ${HARNESS_PORT}..."
# 安全清理旧进程 (防止 set -e 崩溃)
ss -tlnp 2>/dev/null | grep -q ":${HARNESS_PORT} " && \
  fuser -k ${HARNESS_PORT}/tcp 2>/dev/null || true
sleep 1

# 设置上游为 vLLM 的地址
export UPSTREAM_URL="http://localhost:${VLLM_PORT}"
export PROXY_PORT=${HARNESS_PORT}
export PROXY_HOST="0.0.0.0"

# 获取脚本所在目录，定位 claude_harness_proxy.py
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROXY_SCRIPT="${SCRIPT_DIR}/../claude_harness_proxy.py"

nohup $HARNESS_PYTHON $PROXY_SCRIPT > /tmp/claude_harness_proxy.log 2>&1 &

until curl -s http://localhost:${HARNESS_PORT}/health > /dev/null; do
  echo "Waiting for Harness Proxy..."
  sleep 2
done
echo "Harness Proxy is UP!"

echo "=============================================================================="
echo "Gemma-4-31B NVFP4 + Harness Proxy deployed!"
echo "  vLLM (${VLLM_PORT}) -> Harness (${HARNESS_PORT})"
echo "  Final Endpoint: http://localhost:${HARNESS_PORT}/v1"
echo "  Log Files: ${LOG_FILE}, /tmp/claude_harness_proxy.log"
echo "=============================================================================="
