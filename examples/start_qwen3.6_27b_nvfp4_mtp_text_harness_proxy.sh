#!/bin/bash
# Example: Qwen3.6-27B NVFP4 MTP + Harness Proxy
# Description: vLLM -> Harness Proxy chain
#
# This script auto-discovers system configuration and prompts for any
# missing values. Press Enter to accept defaults shown in [brackets].
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────
# Helper: read a value from stdin, with a default
# ──────────────────────────────────────────────────────────────────────────
read_config() {
  local prompt="$1"
  local default="$2"
  if [ -n "$default" ]; then
    read -rp "$prompt [$default] " val
    echo "${val:-$default}"
  else
    read -rp "$prompt " val
    echo "$val"
  fi
}

# ──────────────────────────────────────────────────────────────────────────
# Auto-discovery
# ──────────────────────────────────────────────────────────────────────────

# Detect GPU CUDA compute capability
detect_cuda_arch() {
  local cc
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$cc" ]; then
    echo "${cc}f"
  else
    echo ""
  fi
}

# Find best vLLM-compatible Python from conda environments
detect_python() {
  local candidates=(
    /home/qiba/miniconda/envs/vllm022_py311/bin/python
    /home/qiba/miniconda/envs/vllm021_py311/bin/python
    /home/qiba/miniconda/envs/vllm_nightly_py311/bin/python
    /home/qiba/miniconda/envs/vllm022_py311/bin/python
  )
  for p in "${candidates[@]}"; do
    if [ -x "$p" ]; then
      echo "$p"
      return 0
    fi
  done
  if command -v python3 &>/dev/null; then
    echo "$(command -v python3)"
  else
    echo ""
  fi
}

# Find model directories via filesystem search, let user pick if multiple
find_models() {
  local keyword="$1"
  local search_roots=(
    /home/qiba/ai/models
    /models
    /opt/models
    "${HOME}/models"
  )
  local results=()
  local found_paths=""

  for base in "${search_roots[@]}"; do
    if [ -d "$base" ]; then
      found_paths=$(find "$base" -maxdepth 3 -type d -name "*${keyword}*" 2>/dev/null | sort)
      if [ -n "$found_paths" ]; then
        while IFS= read -r p; do
          results+=("$p")
        done <<< "$found_paths"
      fi
    fi
  done

  echo "${results[@]+"${results[@]}"}"
}

# Prompt user to pick from found models, or type a path
pick_model() {
  local keyword="$1"
  local fallback="$2"
  local candidates
  candidates=$(find_models "$keyword")

  if [ -z "$candidates" ]; then
    echo "$fallback"
    return
  fi

  local arr=()
  while IFS= read -r line; do
    [ -n "$line" ] && arr+=("$line")
  done <<< "$candidates"

  if [ "${#arr[@]}" -eq 1 ]; then
    echo "${arr[0]}"
    return
  fi

  echo "  Found ${#arr[@]} matching model directories:"
  echo ""
  for i in "${!arr[@]}"; do
    echo "    $((i+1))) ${arr[$i]}"
  done
  echo ""
  local choice
  read -rp "Select model directory (1-${#arr[@]}, or type custom path): " choice

  if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#arr[@]}" ]; then
    echo "${arr[$((choice-1))]}"
  elif [ -n "$choice" ] && [ -d "$choice" ]; then
    echo "$choice"
  else
    echo "${arr[0]}"
  fi
}

# ──────────────────────────────────────────────────────────────────────────
# Configuration prompts
# ──────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  Qwen3.6-27B NVFP4 MTP + Harness Proxy Configuration"
echo "============================================================"
echo ""

# Python path
DEFAULT_PYTHON=$(detect_python)
VLLM_PYTHON=$(read_config "Python path" "$DEFAULT_PYTHON")
HARNESS_PYTHON="$VLLM_PYTHON"
if [ ! -x "$VLLM_PYTHON" ]; then
  echo "⚠ Warning: Python not found at $VLLM_PYTHON — proxy may fail to start."
fi

# GPU
DEFAULT_GPU="${CUDA_VISIBLE_DEVICES:-0}"
GPU_DEVICE=$(read_config "GPU device (CUDA_VISIBLE_DEVICES)" "$DEFAULT_GPU")

# CUDA architecture
DEFAULT_CUDA_ARCH=$(detect_cuda_arch)
CUDA_ARCH=$(read_config "CUDA architecture (e.g. 12.0f)" "$DEFAULT_CUDA_ARCH")

# Model path — search filesystem, offer selection if multiple matches
PICKED_MODEL=$(pick_model "Qwen3.6-27B" "/home/qiba/ai/models/Qwen3.6-27B-NVFP4-MTP-TEXT")
MODEL_PATH=$(read_config "Model path" "$PICKED_MODEL")

# Ports
VLLM_PORT=$(read_config "vLLM port" "${VLLM_PORT:-8200}")
HARNESS_PORT=$(read_config "Harness proxy port" "${HARNESS_PORT:-9200}")

# Memory / performance
GPU_MEM_UTIL=$(read_config "GPU memory utilization (0.0-1.0)" "${GPU_MEM_UTIL:-0.82}")
MAX_MODEL_LEN=$(read_config "Max model length" "${MAX_MODEL_LEN:-524288}")
MAX_NUM_SEQS=$(read_config "Max number of sequences" "${MAX_NUM_SEQS:-4}")
MAX_BATCHED_TOKENS=$(read_config "Max batched tokens" "${MAX_BATCHED_TOKENS:-32768}")
BLOCK_SIZE=$(read_config "Block size" "${BLOCK_SIZE:-32}")
MTP_TOKENS=$(read_config "MTP speculative tokens" "${MTP_TOKENS:-4}")

LOG_FILE="/tmp/vllm-harness-qwen36-${VLLM_PORT}.log"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Configuration Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Python:          $VLLM_PYTHON"
echo "  GPU:             $GPU_DEVICE"
echo "  CUDA Arch:       $CUDA_ARCH"
echo "  Model:           $MODEL_PATH"
echo "  vLLM Port:       $VLLM_PORT"
echo "  Harness Port:    $HARNESS_PORT"
echo "  GPU Mem Util:    $GPU_MEM_UTIL"
echo "  Max Model Len:  $MAX_MODEL_LEN"
echo "  MTP Tokens:     $MTP_TOKENS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read -rp "" _

# ──────────────────────────────────────────────────────────────────────────
# Export environment
# ──────────────────────────────────────────────────────────────────────────

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
export FLASHINFER_CUDA_ARCH_LIST="${CUDA_ARCH}"
export FLASHINFER_CACHE_DIR="${HOME}/.cache/flashinfer"
export VLLM_TORCH_COMPILE_CACHE_DIR="${HOME}/.cache/vllm/torch_compile_cache"

# Port check
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
ss -tlnp 2>/dev/null | grep -q ":${HARNESS_PORT} " && \
  fuser -k ${HARNESS_PORT}/tcp 2>/dev/null || true
sleep 1

export UPSTREAM_URL="http://localhost:${VLLM_PORT}"
export PROXY_PORT=${HARNESS_PORT}
export PROXY_HOST="0.0.0.0"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROXY_SCRIPT="${SCRIPT_DIR}/../claude_harness_proxy.py"

nohup $HARNESS_PYTHON $PROXY_SCRIPT > /tmp/claude_harness_proxy.log 2>&1 &

until curl -s http://localhost:${HARNESS_PORT}/health > /dev/null; do
  echo "Waiting for Harness Proxy..."
  sleep 2
done
echo "Harness Proxy is UP!"

echo ""
echo "=============================================================================="
echo "  Qwen3.6-27B NVFP4 MTP + Harness Proxy deployed!"
echo "  vLLM (${VLLM_PORT}) -> Harness (${HARNESS_PORT})"
echo "  Final Endpoint: http://localhost:${HARNESS_PORT}/v1"
echo "  Logs: ${LOG_FILE}, /tmp/claude_harness_proxy.log"
echo "=============================================================================="
