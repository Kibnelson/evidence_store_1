#!/usr/bin/env bash
# Build the historical in-process Full Blend runtime in node-local scratch.
set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run inside the four-GPU Slurm allocation}"
test "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}" = 1
if (( $# > 1 )); then
  printf 'Usage: %s [activation-receipt]\n' "$0" >&2
  exit 2
fi
activation_receipt="${1:-}"
if [[ -n "$activation_receipt" ]]; then
  activation_receipt="$(cd "$(dirname -- "$activation_receipt")" && pwd -P)/$(basename -- "$activation_receipt")"
fi
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE VLLM_USE_V2_MODEL_RUNNER TRANSFORMERS_CACHE
unset VLLM_USE_PRECOMPILED VLLM_USE_PRECOMPILED_RUST VLLM_PRECOMPILED_WHEEL_LOCATION \
  VLLM_PRECOMPILED_WHEEL_COMMIT VLLM_PRECOMPILED_WHEEL_VARIANT SCHED_V2_PRECOMPILED_WHEEL_RECEIPT
unset "${!SCHED_V2_@}" "${!LMCACHE_@}" ES_BLEND_TRACE
export PYTHONNOUSERSITE=1

PD_HOME_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source_commit="$(git -C "$PD_HOME_REPO" rev-parse HEAD)"
node_root="/tmp/$(id -un)"
mkdir -p "$node_root"
if [[ ! -O "$node_root" || -L "$node_root" ]]; then
  printf 'Expected an owned node-local directory at %s\n' "$node_root" >&2
  exit 1
fi
PD_EXEC_ROOT="$(mktemp -d "$node_root/es-legacy-blend-$SLURM_JOB_ID.XXXXXX")"
REPO="$PD_EXEC_ROOT/repo"
CHECK_ROOT="$PD_EXEC_ROOT/check"
PD_ENV="$PD_EXEC_ROOT/venv"
mkdir -p "$REPO" "$CHECK_ROOT"

(
  printf 'Runtime scratch: %s\nExperiment source: %s\n' "$PD_EXEC_ROOT" "$source_commit"
  printf '%s\n' "$source_commit" > "$CHECK_ROOT/experiment-source.txt"

  export UV_CACHE_DIR="$node_root/uv-cache" PIP_CACHE_DIR="$node_root/pip-cache"
  export UV_PYTHON_INSTALL_DIR="$PD_EXEC_ROOT/uv-python" UV_PYTHON_CACHE_DIR="$PD_EXEC_ROOT/uv-python-cache"
  export HF_HOME="$node_root/huggingface"
  export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
  export XDG_CACHE_HOME="$PD_EXEC_ROOT/xdg-cache"
  export TMPDIR="$PD_EXEC_ROOT/tmp" VLLM_RPC_BASE_PATH="$PD_EXEC_ROOT/rpc"
  export TORCH_EXTENSIONS_DIR="$PD_EXEC_ROOT/torch-extensions"
  export TRITON_CACHE_DIR="$PD_EXEC_ROOT/triton" CUDA_CACHE_PATH="$PD_EXEC_ROOT/cuda-cache"
  mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_PYTHON_CACHE_DIR" \
    "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" "$VLLM_RPC_BASE_PATH"
  df -h "$PD_EXEC_ROOT"
  command -v uv curl nvidia-smi
  nvidia-smi --query-gpu=uuid,name,driver_version --format=csv

  # Job 29538 used the site's CUDA 12 compiler and the cu129 Torch/vLLM libraries.
  CUDA_HOME="${EVIDENCESTORE_LEGACY_CUDA_HOME:-}"
  if [[ -z "$CUDA_HOME" ]] && nvcc="$(command -v nvcc)"; then
    CUDA_HOME="$(dirname "$(dirname "$(readlink -f -- "$nvcc")")")"
  fi
  if [[ -z "$CUDA_HOME" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    printf 'Load the site CUDA 12 toolkit or set EVIDENCESTORE_LEGACY_CUDA_HOME to it.\n' >&2
    exit 1
  fi
  cuda_info="$("$CUDA_HOME/bin/nvcc" --version)"
  printf 'CUDA toolkit: %s\n%s\n' "$CUDA_HOME" "$cuda_info" | tee "$CHECK_ROOT/cuda-toolkit.txt"
  if [[ "$cuda_info" != *'release 12.'* ]]; then
    printf 'Legacy Full Blend requires a CUDA 12 compiler. Set EVIDENCESTORE_LEGACY_CUDA_HOME.\n' >&2
    exit 1
  fi
  export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" VLLM_MAIN_CUDA_VERSION=12.9
  export CC="${CC:-gcc}" CXX="${CXX:-g++}"
  command -v "$CC" "$CXX"
  export MAX_JOBS="${MAX_JOBS:-4}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

  uv --no-config python install --no-bin 3.12.14
  base_python="$(env -u VIRTUAL_ENV -u CONDA_PREFIX uv --no-config --offline python find \
    --managed-python --no-python-downloads 3.12.14)"
  base_python="$(readlink -f -- "$base_python")"
  if [[ "$base_python" != "$(readlink -f -- "$UV_PYTHON_INSTALL_DIR")/"* || ! -x "$base_python" ]]; then
    printf 'uv selected Python outside this runtime scratch directory: %s\n' "$base_python" >&2
    exit 1
  fi
  uv --no-config venv --python "$base_python" --no-python-downloads "$PD_ENV"
  PYTHON="$PD_ENV/bin/python"
  "$PYTHON" -I - "$UV_PYTHON_INSTALL_DIR" <<'PY' | tee "$CHECK_ROOT/python.txt"
import encodings
from pathlib import Path
import ssl
import sys
import sysconfig

root = Path(sys.argv[1]).resolve()
if (sys.version_info[:3] != (3, 12, 14)
        or not Path(sys.base_prefix).resolve().is_relative_to(root)
        or not Path(encodings.__file__).resolve().is_relative_to(root)
        or not (Path(sysconfig.get_path("include")) / "Python.h").is_file()):
    raise SystemExit("Expected Python 3.12.14 and its headers in this runtime scratch directory")
print(f"Managed Python verified: {sys.version.split()[0]} at {sys.executable}")
PY

  git -C "$PD_HOME_REPO" archive "$source_commit" | tar -xf - -C "$REPO"
  cd "$REPO"
  "$PYTHON" scripts/checkout_sources.py vllm-legacy-base lmcache-legacy-base faiss
  "$PYTHON" scripts/vllm_pd_prepare.py --legacy-full-blend
  "$PYTHON" scripts/lmcache_pd_prepare.py --legacy-full-blend
  VLLM_SNAPSHOT="$REPO/third_party/forks/vllm-full-blend"
  LMCACHE_SNAPSHOT="$REPO/third_party/forks/lmcache-full-blend"
  VLLM_SOURCE="$PD_EXEC_ROOT/build/vllm"
  LMCACHE_SOURCE="$PD_EXEC_ROOT/build/lmcache"
  mkdir -p "$VLLM_SOURCE" "$LMCACHE_SOURCE"
  git -C "$VLLM_SNAPSHOT" archive HEAD | tar -xf - -C "$VLLM_SOURCE"
  git -C "$LMCACHE_SNAPSHOT" archive HEAD | tar -xf - -C "$LMCACHE_SOURCE"
  standalone_patch="$REPO/third_party/legacy-full-blend-standalone.patch"
  lane_support_patch="$REPO/third_party/legacy-full-blend-lane-support.patch"
  git -C "$PD_EXEC_ROOT/build" apply "$standalone_patch"
  git -C "$PD_EXEC_ROOT/build" apply "$lane_support_patch"
  "$PYTHON" scripts/vllm_pd_legacy_trace_overlay.py --vllm-source "$VLLM_SOURCE" \
    --lmcache-source "$LMCACHE_SOURCE" | tee "$CHECK_ROOT/trace-overlay.json"
  vllm_base="$(git -C "$REPO/third_party/checkouts/vllm-legacy-base" rev-parse HEAD)"
  architecture="$(uname -m)"
  wheel_url="https://wheels.vllm.ai/$vllm_base/vllm-0.23.0%2Bcu129-cp38-abi3-manylinux_2_28_$architecture.whl"
  wheel_receipt="$CHECK_ROOT/vllm-precompiled-wheel.json"

  # Preserve the compatible dependency family used with the historical editable builds.
  constraints="$CHECK_ROOT/legacy-runtime-constraints.txt"
  cat > "$constraints" <<'CONSTRAINTS'
torch==2.11.0+cu129
cuda-python==12.9.7
cuda-bindings==12.9.7
transformers==5.17.0
xgrammar==0.2.3
apache-tvm-ffi==0.1.9
tilelang==0.1.9
nixl==1.4.1
opentelemetry-api==1.40.0
opentelemetry-sdk==1.40.0
opentelemetry-exporter-otlp==1.40.0
opentelemetry-exporter-otlp-proto-common==1.40.0
opentelemetry-exporter-otlp-proto-grpc==1.40.0
opentelemetry-exporter-otlp-proto-http==1.40.0
opentelemetry-exporter-prometheus==0.61b0
opentelemetry-proto==1.40.0
opentelemetry-semantic-conventions==0.61b0
opentelemetry-semantic-conventions-ai==0.5.1
CONSTRAINTS

  install=(uv --no-config pip install --python "$PYTHON" --constraint "$constraints" --torch-backend=cu129)
  "${install[@]}" -r "$VLLM_SOURCE/requirements/build/cuda.txt"
  VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$wheel_url" \
    VLLM_PRECOMPILED_WHEEL_COMMIT="$vllm_base" SCHED_V2_PRECOMPILED_WHEEL_RECEIPT="$wheel_receipt" \
    VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX=1 VLLM_VERSION_OVERRIDE=0.23.0 \
    "${install[@]}" --no-build-isolation --editable "$VLLM_SOURCE"
  "${install[@]}" -r "$LMCACHE_SOURCE/requirements/build.txt"
  BUILD_WITH_CUDA=1 NO_GPU_EXT=0 LMCACHE_CUDA_MAJOR=12 \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.5.0 SETUPTOOLS_SCM_PRETEND_VERSION=0.5.0 \
    "${install[@]}" --no-build-isolation --editable "$LMCACHE_SOURCE"
  "${install[@]}" 'nixl==1.4.1' 'cuda-python==12.9.7' 'cuda-bindings==12.9.7' faiss-cpu
  uv --no-config pip check --python "$PYTHON"
  uv --no-config pip freeze --python "$PYTHON" > "$CHECK_ROOT/dependencies.txt"

  EVIDENCESTORE_PD_TRACE=0 "$PYTHON" - "$VLLM_SOURCE" "$LMCACHE_SOURCE" "$vllm_base" \
    "$CHECK_ROOT/trace-overlay.json" "$wheel_receipt" "$standalone_patch" "$lane_support_patch" \
    "$CHECK_ROOT/runtime.json" <<'PY'
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

import faiss
import torch
import vllm
from vllm import _C
import lmcache
from lmcache import c_ops
from nixl._api import nixl_agent, nixl_agent_config
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

(vllm_source, lmcache_source, wheel_base, trace_overlay, wheel_receipt,
 standalone_patch, lane_support_patch, runtime_receipt) = sys.argv[1:]
expected = {"vllm": "0.23.0", "lmcache": "0.5.0", "torch": "2.11.0+cu129", "nixl": "1.4.1"}
observed = {name: version(name) for name in expected}
wheel = json.loads(Path(wheel_receipt).read_text())
if (wheel["variant"] != "cu129" or wheel["precompiled_commit"] != wheel_base
        or wheel["source_root"] != vllm_source or wheel["slurm_job_id"] != os.environ["SLURM_JOB_ID"]):
    raise SystemExit("The vLLM wheel receipt does not match this runtime and allocation")
if observed != expected or torch.version.cuda != "12.9" or torch.cuda.device_count() != 4:
    raise SystemExit(f"Expected the historical CUDA 12.9 runtime on four visible GPUs, got {observed}")
for module, path in ((vllm, vllm_source), (lmcache, lmcache_source)):
    if not Path(module.__file__).resolve().is_relative_to(Path(path).resolve()):
        raise SystemExit(f"{module.__name__} did not load from the restored source at {path}")
for name in ("NixlConnector", "LMCacheConnectorV1"):
    connector = KVConnectorFactory.get_connector_class(KVTransferConfig(kv_connector=name, kv_role="kv_both"))
    if connector.__name__ != name:
        raise SystemExit(f"Unexpected connector factory result: {connector}")
agent = nixl_agent(f"legacy-blend-preflight-{os.getpid()}", nixl_agent_config(backends=[]))
if "UCX" not in {str(item) for item in agent.get_plugin_list()}:
    raise SystemExit("NIXL cannot load UCX")
agent.create_backend("UCX", {})
memory_types = sorted(str(item) for item in agent.get_backend_mem_types("UCX"))
if not any("VRAM" in item.upper() for item in memory_types):
    raise SystemExit(f"NIXL UCX does not advertise GPU memory: {memory_types}")
receipt = {**observed, "faiss": faiss.__version__, "vllm_wheel_base": wheel_base,
           "torch_cuda": torch.version.cuda, "ucx_memory_types": memory_types,
           "vllm_native": _C.__file__, "lmcache_native": c_ops.__file__,
           "worker_runner_module": GPUModelRunner.__module__,
           "vllm_wheel_sha256": wheel["wheel_sha256"],
           "standalone_patch_sha256": hashlib.sha256(Path(standalone_patch).read_bytes()).hexdigest(),
           "lane_support_patch_sha256": hashlib.sha256(Path(lane_support_patch).read_bytes()).hexdigest(),
           "trace_overlay_sha256": hashlib.sha256(Path(trace_overlay).read_bytes()).hexdigest()}
encoded = json.dumps(receipt, indent=2)
Path(runtime_receipt).write_text(encoded + "\n", encoding="utf-8")
print(encoded)
PY

  activation="$PD_EXEC_ROOT/activate.sh"
  {
    printf 'unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE VLLM_USE_V2_MODEL_RUNNER TRANSFORMERS_CACHE\n'
    printf 'unset "${!SCHED_V2_@}" "${!LMCACHE_@}" ES_BLEND_TRACE\n'
    for name in PD_HOME_REPO PD_EXEC_ROOT REPO CHECK_ROOT PD_ENV PYTHON PYTHONNOUSERSITE \
      VLLM_SOURCE LMCACHE_SOURCE VLLM_SNAPSHOT LMCACHE_SNAPSHOT \
      UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_PYTHON_CACHE_DIR PIP_CACHE_DIR HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE \
      XDG_CACHE_HOME TMPDIR VLLM_RPC_BASE_PATH TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR CUDA_CACHE_PATH \
      CUDA_HOME VLLM_MAIN_CUDA_VERSION CC CXX MAX_JOBS TORCH_CUDA_ARCH_LIST; do
      printf 'export %s=%q\n' "$name" "${!name}"
    done
    printf 'source %q && export PATH=%q:"$PATH" && cd %q\n' "$PD_ENV/bin/activate" "$CUDA_HOME/bin" "$REPO"
  } > "$activation"
  if [[ -n "$activation_receipt" ]]; then
    temp_receipt="$(mktemp "$activation_receipt.XXXXXX")"
    printf '%s\n' "$activation" > "$temp_receipt"
    mv -f -- "$temp_receipt" "$activation_receipt"
  fi
  printf '\nLegacy Full Blend runtime verified. Activate it in the allocation shell:\n  source %q\n' "$activation"
) 2>&1 | tee "$CHECK_ROOT/setup.log"
