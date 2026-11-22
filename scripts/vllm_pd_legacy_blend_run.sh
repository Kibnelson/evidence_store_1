#!/usr/bin/env bash
# Build, replay, and save the public in-process Full Blend smoke in one allocation.
set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run the Full Blend smoke in the four-GPU allocation}"
layout="${1:-1b3d}"
trace="${2:-on}"
workload="${3:-musique-smoke}"
prompt_policy="${4:-default}"
estimate_delta=0
trace_kv_storage=0
trace_kv_pages=0
capture_kv_pages=0
for option in "${@:5}"; do
  case "$option" in
    --estimate-delta) estimate_delta=1 ;;
    --trace-kv-storage) trace_kv_storage=1 ;;
    --trace-kv-pages) trace_kv_pages=1 ;;
    --capture-kv-pages) capture_kv_pages=1 ;;
    *) printf 'Unknown Full Blend option %s\n' "$option" >&2; exit 2 ;;
  esac
done
case "$layout" in
  1b3d) topology=1p3d ;;
  2b2d) topology=2p2d ;;
  both) topology=both ;;
  *) printf 'Layout must be 1b3d, 2b2d, or both.\n' >&2; exit 2 ;;
esac
case "$workload" in
  musique-smoke) run_prefix=musique ;;
  musique-single-turn-40|mtrag-multi-turn-repeat) run_prefix="$workload" ;;
  *) printf 'Unknown Full Blend workload %s\n' "$workload" >&2; exit 2 ;;
esac
if [[ "$trace" != on && "$trace" != off ]]; then
  printf 'Usage: %s [1b3d|2b2d|both] [on|off] [workload] [prompt-policy] [--estimate-delta] [--trace-kv-storage|--trace-kv-pages] [--capture-kv-pages]\n' "$0" >&2
  exit 2
fi
case "$prompt_policy:$workload" in
  default:*|seen-first:musique-*|openclaw:mtrag-multi-turn-repeat|previous-turn-first:mtrag-multi-turn-repeat) ;;
  *) printf 'Use seen-first with MuSiQue, and openclaw or previous-turn-first with MT-RAG.\n' >&2; exit 2 ;;
esac
if (( estimate_delta )); then
  case "$trace:$prompt_policy:$workload" in
    on:previous-turn-first:mtrag-multi-turn-repeat|on:seen-first:musique-single-turn-40) ;;
    *) printf 'Use --estimate-delta with trace on and MT-RAG previous-turn-first or MuSiQue seen-first.\n' >&2; exit 2 ;;
  esac
fi
if (( trace_kv_storage )) && [[ "$trace:$workload" != on:mtrag-multi-turn-repeat ]]; then
  printf 'Use --trace-kv-storage with trace on and mtrag-multi-turn-repeat.\n' >&2
  exit 2
fi
if (( trace_kv_pages )) && { [[ "$layout:$trace:$workload:$prompt_policy" != 1b3d:on:mtrag-multi-turn-repeat:previous-turn-first ]] ||
    (( trace_kv_storage || estimate_delta )); }; then
  printf 'Use --trace-kv-pages with 1b3d on mtrag-multi-turn-repeat previous-turn-first.\n' >&2
  exit 2
fi
if (( capture_kv_pages && !trace_kv_pages )); then
  printf 'Use --capture-kv-pages with --trace-kv-pages.\n' >&2
  exit 2
fi

source_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
node_root="/tmp/$(id -un)"
mkdir -p "$node_root"
[[ -O "$node_root" && ! -L "$node_root" ]]
activation_receipt=""
archive=""
trap 'rm -f -- "$activation_receipt" "$archive"' EXIT
retained="${EVIDENCESTORE_LEGACY_RUNTIME:-}"
if [[ -n "$retained" ]]; then
  if [[ "$retained" != "$node_root/"* || ! -O "$retained" || -L "$retained" ||
        ! -r "$retained/activate.sh" || ! -s "$retained/check/runtime.json" ]]; then
    printf 'Expected a verified Full Blend runtime in this account’s scratch directory.\n' >&2
    exit 1
  fi
  source "$retained/activate.sh"
  [[ "$retained" == "$PD_EXEC_ROOT" ]]
  launcher_repo="$source_repo"
  for file in third_party/{vllm,lmcache}-full-blend-fork.json \
      third_party/legacy-full-blend-standalone.patch \
      scripts/vllm_pd_legacy_trace_{events,overlay}.py \
      scripts/vllm_pd_legacy_storage_trace.py scripts/vllm_pd_legacy_page_trace.py; do
    if ! cmp -s "$source_repo/$file" "$REPO/$file"; then
      printf 'The retained runtime uses a different %s. Build a fresh runtime.\n' "$file" >&2
      exit 1
    fi
  done
  lane_patch="$source_repo/third_party/legacy-full-blend-lane-support.patch"
  if git -C "$PD_EXEC_ROOT/build" apply --check "$lane_patch" 2>/dev/null; then
    git -C "$PD_EXEC_ROOT/build" apply "$lane_patch"
  else
    git -C "$PD_EXEC_ROOT/build" apply --reverse --check "$lane_patch"
  fi
  EVIDENCESTORE_PD_TRACE=0 "$PYTHON" -c \
    'from vllm.v1.worker.gpu_model_runner import GPUModelRunner; print("Legacy GPUModelRunner verified")'
  printf 'Reusing Full Blend runtime %s\n' "$PD_EXEC_ROOT"
else
  activation_receipt="$(mktemp "$node_root/legacy-blend-activation-$SLURM_JOB_ID.XXXXXX")"
  bash "$source_repo/scripts/vllm_pd_legacy_blend_runtime.sh" "$activation_receipt"
  source "$(<"$activation_receipt")"
  launcher_repo="$REPO"
fi

replay="$launcher_repo/scripts/vllm_pd_replay.py"
if [[ "$workload" == mtrag-multi-turn-repeat ]]; then
  domain="$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["domain"])' \
    "$launcher_repo/workloads/$workload.json")"
  [[ "$domain" =~ ^(clapnq|cloud|fiqa|govt)$ ]]
  prepared_override="${MTRAG_PREPARED:-}"
  prepared="${prepared_override:-$PD_EXEC_ROOT/data/mtrag/$domain}"
  if [[ -z "$prepared_override" && ! -s "$prepared/corpus.jsonl" ]]; then
    "$PYTHON" "$launcher_repo/scripts/checkout_sources.py" mt-rag
    "$PYTHON" "$replay" prepare-data --dataset mtrag --source "$launcher_repo/third_party/checkouts/mt-rag" \
      --output "$PD_EXEC_ROOT/data/mtrag"
  fi
else
  prepared_override="${MUSIQUE_PREPARED:-}"
  prepared="${prepared_override:-$PD_EXEC_ROOT/data/musique}"
  if [[ -z "$prepared_override" && ! -s "$prepared/corpus.jsonl" ]]; then
    musique_dev="$("$PYTHON" "$launcher_repo/scripts/vllm_pd_stage_inputs.py" musique "$PD_EXEC_ROOT/data/inputs")"
    "$PYTHON" "$replay" prepare-data --dataset musique --source "$musique_dev" --output "$prepared"
  fi
fi
if [[ -z "$prepared_override" && ! -s "$prepared/index/manifest.json" ]]; then
  "$PYTHON" "$replay" build-index --corpus "$prepared/corpus.jsonl" --index-dir "$prepared/index" \
    --embedding-model intfloat/e5-base-v2 --embedding-revision f52bf8ec8c7124536f0efb74aca902b2995e5bcd \
    --embedding-device cuda:0
fi

run_name="$run_prefix-legacy-full-blend-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
run_root="$REPO/reports/runs/$run_name"
run_log="$PD_EXEC_ROOT/smoke.log"
if [[ -n "$retained" ]]; then
  run_log="$PD_EXEC_ROOT/$run_name.log"
  git -C "$source_repo" rev-parse HEAD > "$CHECK_ROOT/$run_name-replay-source.txt"
  sha256sum "$lane_patch" > "$CHECK_ROOT/$run_name-lane-support.sha256"
fi
printf 'Run artifacts: %s\n' "$run_root"
replay_args=(run --topology "$topology" --trace "$trace" --workload "$workload"
  --prompt-policy "$prompt_policy" --prepared "$prepared" --run-root "$run_root")
if (( estimate_delta )); then
  replay_args+=(--estimate-delta)
fi
if (( trace_kv_storage )); then
  replay_args+=(--trace-kv-storage)
fi
if (( trace_kv_pages )); then
  replay_args+=(--trace-kv-pages)
fi
if (( capture_kv_pages )); then
  replay_args+=(--capture-kv-pages)
fi
"$PYTHON" "$launcher_repo/scripts/vllm_pd_legacy_blend.py" "${replay_args[@]}" 2>&1 | tee "$run_log"

report_dir="${EVIDENCESTORE_LEGACY_REPORT_DIR:-$HOME/evidence_store_legacy_reports}"
mkdir -p "$report_dir"
archive="$(mktemp "$report_dir/.${run_name}.XXXXXX")"
tar -czf "$archive" -C "$PD_EXEC_ROOT" check "${run_log##*/}" -C "$REPO/reports/runs" "$run_name"
tar -tzf "$archive" >/dev/null
mv -- "$archive" "$report_dir/$run_name.tar.gz"
sha256sum "$report_dir/$run_name.tar.gz"
if (( trace_kv_pages )); then
  printf 'MT-RAG KV page trace archived at %s\n' "$report_dir/$run_name.tar.gz"
elif (( trace_kv_storage )); then
  printf 'MT-RAG KV storage trace archived at %s\n' "$report_dir/$run_name.tar.gz"
else
  printf 'Legacy Full Blend smoke passed. Archive: %s\n' "$report_dir/$run_name.tar.gz"
fi
