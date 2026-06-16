#!/bin/bash
# Example: Qwen3.6-27B NVFP4 MTP + Harness Proxy
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
VLLM_PORT="${VLLM_PORT:-8200}"
HARNESS_PORT="${HARNESS_PORT:-9200}"

# ── 环境变量 ──────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512"
export TORCH_CUDA_ALLOW_TF32=0
export TORCH_BACKEND_CUDNN_ALLOW_TF32=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_SLEEP_WHEN_IDLE="1"
export VLLM_ENABLE_CUDAGRAPH_GC="1"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
export RAY_memory_monitor_refresh_ms="0"
export OMP_NUM_THREADS="1"
export PYTHONIOENCODING="utf-8"
export USE_LIBUV="0"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="0"
export NCCL_ASYNC_ERROR_HANDLING="0"
export PYTHONFAULTHANDLER="1"
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export FLASHINFER_CACHE_DIR="${HOME}/.cache/flashinfer"
export VLLM_TORCH_COMPILE_CACHE_DIR="${HOME}/.cache/vllm/torch_compile_cache"
# ──────────────────────────────────────────────────────────────────────────────

# ── 模型配置 ──────────────────────────────────────────────────────────────────
# 修改为你的模型路径
MODEL_PATH="${MODEL_PATH:-/home/qiba/ai/models/Qwen3.6-27B-NVFP4-MTP-TEXT}"
LOG_FILE="/tmp/vllm-harness-qwen36-${VLLM_PORT}.log"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.82}"
MAX_MODEL_LEN=524288
MAX_NUM_SEQS=4
MAX_BATCHED_TOKENS=32768
BLOCK_SIZE=32
MTP_TOKENS=4

if ss -tlnp 2>/dev/null | grep -q ":${VLLM_PORT} "; then
  echo "错误：vLLM 端口 ${VLLM_PORT} 已被占用"; exit 1
fi

# 1. 预编译 FlashInfer 内核
echo "Step 0: Precompiling FlashInfer kernels..."
mkdir -p "${FLASHINFER_CACHE_DIR}" "${VLLM_TORCH_COMPILE_CACHE_DIR}"

"${VLLM_PYTHON}" -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '${GPU_DEVICE}'
import torch
from flashinfer.attention import BatchAttention
_ = BatchAttention()
try:
    from flashinfer.fused_moe import fused_moe
except Exception:
    pass
from flashinfer import norm, rope
print('FlashInfer precompile done')
" 2>&1 || echo "FlashInfer precompile warning (non-fatal)"

# 2. 启动 vLLM 引擎
echo "Step 1: Starting vLLM Engine on port ${VLLM_PORT}..."
> "${LOG_FILE}"
nohup "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name qwen3.6-27b \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"preserve_thinking": false, "enable_thinking": false}' \
  --host 0.0.0.0 --port "${VLLM_PORT}" \
  --attention-backend FLASHINFER \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
  --block-size "${BLOCK_SIZE}" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --quantization compressed-tensors \
  --speculative-config '{"method":"mtp","num_speculative_tokens":'"${MTP_TOKENS}"',"draft_sample_method":"probabilistic"}' \
  --kv-cache-dtype auto \
  --dtype auto \
  --tokenizer-mode auto \
  --no-use-tqdm-on-load \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  >> "${LOG_FILE}" 2>&1 &

# 等待 vLLM 就绪
until curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null; do
  echo "Waiting for vLLM to initialize..."
  sleep 5
done
echo "vLLM Engine is UP!"

# 3. 启动 Claude Harness Proxy
echo "Step 2: Starting Claude Harness Proxy on port ${HARNESS_PORT}..."
# 只杀同端口的旧 proxy 实例，不影响其他端口
ss -tlnp 2>/dev/null | grep -q ":${HARNESS_PORT} " && \
  fuser -k ${HARNESS_PORT}/tcp 2>/dev/null || true
sleep 1

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
echo "Qwen3.6-27B NVFP4 MTP + Harness Proxy deployed!"
echo "  vLLM (${VLLM_PORT}) -> Harness (${HARNESS_PORT})"
echo "  Final Endpoint: http://localhost:${HARNESS_PORT}/v1"
echo "  Logs: ${LOG_FILE}, /tmp/claude_harness_proxy.log"
echo "=============================================================================="
