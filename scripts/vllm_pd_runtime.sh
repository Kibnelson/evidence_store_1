#!/usr/bin/env bash
set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run this script inside the four-GPU Slurm allocation}"
test "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}" = 1
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE VLLM_USE_V2_MODEL_RUNNER TRANSFORMERS_CACHE
unset VLLM_USE_PRECOMPILED VLLM_USE_PRECOMPILED_RUST VLLM_PRECOMPILED_WHEEL_LOCATION \
  VLLM_PRECOMPILED_WHEEL_COMMIT VLLM_PRECOMPILED_WHEEL_VARIANT
export PYTHONNOUSERSITE=1

PD_HOME_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source_commit="$(git -C "$PD_HOME_REPO" rev-parse HEAD)"
node_root="/tmp/$(id -un)"
mkdir -p "$node_root"
if [[ ! -O "$node_root" || -L "$node_root" ]]; then
  printf 'Expected a node-local directory owned by this account at %s\n' "$node_root" >&2
  exit 1
fi
PD_EXEC_ROOT="$(mktemp -d "$node_root/espd-$SLURM_JOB_ID.XXXXXX")"
REPO="$PD_EXEC_ROOT/repo"
CHECK_ROOT="$PD_EXEC_ROOT/check"
PD_ENV="$PD_EXEC_ROOT/venv"
mkdir -p "$REPO" "$CHECK_ROOT"
(
  printf 'Runtime scratch: %s\nExperiment source: %s\n' "$PD_EXEC_ROOT" "$source_commit"
  printf '%s\n' "$source_commit" > "$CHECK_ROOT/experiment-source.txt"

  export UV_CACHE_DIR="$node_root/uv-cache"
  export UV_PYTHON_INSTALL_DIR="$node_root/uv-python"
  export PIP_CACHE_DIR="$node_root/pip-cache"
  export HF_HOME="$node_root/huggingface"
  export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
  export XDG_CACHE_HOME="$PD_EXEC_ROOT/xdg-cache"
  export TMPDIR="$PD_EXEC_ROOT/tmp" VLLM_RPC_BASE_PATH="$PD_EXEC_ROOT/rpc"
  export TORCH_EXTENSIONS_DIR="$PD_EXEC_ROOT/torch-extensions"
  export TRITON_CACHE_DIR="$PD_EXEC_ROOT/triton" CUDA_CACHE_PATH="$PD_EXEC_ROOT/cuda-cache"
  export CCACHE_DIR="$PD_EXEC_ROOT/ccache" SCCACHE_DIR="$PD_EXEC_ROOT/sccache"
  mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$HF_HUB_CACHE" \
    "$XDG_CACHE_HOME" "$TMPDIR" "$VLLM_RPC_BASE_PATH"
  df -h "$PD_EXEC_ROOT"
  df -i "$PD_EXEC_ROOT"

  command -v uv nvcc nvidia-smi
  export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")}"
  export PATH="$CUDA_HOME/bin:$PATH"
  if [[ -z "${CC:-}${CXX:-}" ]] && command -v gcc-13 >/dev/null && command -v g++-13 >/dev/null; then
    CC=gcc-13
    CXX=g++-13
  fi
  export CC="${CC:-gcc}" CXX="${CXX:-g++}"
  command -v "$CC" "$CXX"
  compiler_version="$("$CXX" -dumpfullversion -dumpversion)"
  printf 'Host compiler: %s %s\n' "$CXX" "$compiler_version"
  cuda_info="$("$CUDA_HOME/bin/nvcc" --version)"
  printf '%s\n' "$cuda_info"
  if [[ "$cuda_info" != *'release 13.'* ]]; then
    printf 'This runtime uses the pinned CUDA 13.2 Torch build. Select a CUDA 13 toolkit such as 13.2 with CUDA_HOME before building LMCache.\n' >&2
    exit 1
  fi
  nvidia-smi --query-gpu=uuid,name,driver_version --format=csv
  export MAX_JOBS="${MAX_JOBS:-4}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

  git -C "$PD_HOME_REPO" archive "$source_commit" | tar -xf - -C "$REPO"
  cd "$REPO"
  python3 scripts/checkout_sources.py vllm lmcache faiss mt-rag
  python3 scripts/vllm_pd_prepare.py
  VLLM_SOURCE="$REPO/third_party/forks/vllm"
  LMCACHE_SOURCE="$REPO/third_party/checkouts/lmcache"
  vllm_base="$(python3 -c 'import json; print(json.load(open("third_party/vllm-fork.json"))["base_commit"])')"
  vllm_build=source
  if [[ "$cuda_info" == *'release 13.2,'* ]] &&
      git -C "$VLLM_SOURCE" diff --quiet "$vllm_base" HEAD -- . ':(exclude,glob)vllm/**/*.py'; then
    vllm_build=precompiled-cu132
  elif [[ ! "$compiler_version" =~ ^[0-9]+(\.[0-9]+)*$ ]] || (( ${compiler_version%%.*} < 13 )); then
    printf 'The vLLM CUDA source build needs GCC/G++ 13 or later. %s reports %s. Set CC and CXX to a compatible pair.\n' \
      "$CXX" "$compiler_version" >&2
    exit 1
  fi
  printf 'vLLM build: %s (upstream %s)\n' "$vllm_build" "$vllm_base" | tee "$CHECK_ROOT/vllm-build.txt"

  uv cache dir
  uv venv --python 3.12.14 "$PD_ENV"
  PYTHON="$PD_ENV/bin/python"
  if [[ "$vllm_build" == precompiled-cu132 ]]; then
    VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT="$vllm_base" VLLM_PRECOMPILED_WHEEL_VARIANT=cu132 \
      uv pip install --python "$PYTHON" --torch-backend=cu132 --editable "$VLLM_SOURCE"
    "$PD_ENV/bin/vllm" download-kernels
  else
    uv pip install --python "$PYTHON" --torch-backend=cu132 --editable "$VLLM_SOURCE"
  fi
  uv pip install --python "$PYTHON" 'nixl==1.5.2' faiss-cpu
  uv pip install --python "$PYTHON" -r "$LMCACHE_SOURCE/requirements/build.txt"
  uv pip install --python "$PYTHON" --no-build-isolation --editable "$LMCACHE_SOURCE"

  "$PYTHON" - <<'PY' | tee "$CHECK_ROOT/runtime.json"
from importlib.metadata import version
import json
import os

import torch
from nixl._api import nixl_agent, nixl_agent_config

if torch.cuda.device_count() != 4 or not torch.version.cuda:
    raise SystemExit("Expected a CUDA build with exactly four visible GPUs")
agent = nixl_agent(f"pd-preflight-{os.getpid()}", nixl_agent_config(backends=[]))
if "UCX" not in {str(item) for item in agent.get_plugin_list()}:
    raise SystemExit("NIXL cannot load the UCX backend")
agent.create_backend("UCX", {})
memory_types = sorted(str(item) for item in agent.get_backend_mem_types("UCX"))
if not any("VRAM" in item.upper() for item in memory_types):
    raise SystemExit(f"NIXL UCX does not advertise GPU memory: {memory_types}")
print(json.dumps({"vllm": version("vllm"), "nixl": version("nixl"),
                  "lmcache": version("lmcache"), "torch": version("torch"),
                  "torch_cuda": torch.version.cuda, "ucx_memory_types": memory_types}, indent=2))
PY

  activation="$PD_EXEC_ROOT/activate.sh"
  {
    printf 'unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE VLLM_USE_V2_MODEL_RUNNER TRANSFORMERS_CACHE\n'
    for name in PD_HOME_REPO PD_EXEC_ROOT REPO CHECK_ROOT PD_ENV PYTHON PYTHONNOUSERSITE \
      UV_CACHE_DIR UV_PYTHON_INSTALL_DIR PIP_CACHE_DIR HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE \
      XDG_CACHE_HOME TMPDIR VLLM_RPC_BASE_PATH TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR CUDA_CACHE_PATH \
      CCACHE_DIR SCCACHE_DIR CUDA_HOME CC CXX MAX_JOBS TORCH_CUDA_ARCH_LIST; do
      printf 'export %s=%q\n' "$name" "${!name}"
    done
    printf 'export PATH=%q:"$PATH"\n' "$CUDA_HOME/bin"
    printf 'source %q && cd %q && set -euo pipefail\n' "$PD_ENV/bin/activate" "$REPO"
  } > "$activation"
  printf '\nRuntime verified. Activate it in the allocation shell:\n  source %q\n' "$activation"
) 2>&1 | tee "$CHECK_ROOT/setup.log"
