#!/usr/bin/env bash
# Build, replay, and save the public in-process Full Blend smoke in one allocation.
set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run the Full Blend smoke in the four-GPU allocation}"
layout="${1:-1b3d}"
trace="${2:-on}"
case "$layout" in
  1b3d) topology=1p3d ;;
  2b2d) topology=2p2d ;;
  both) topology=both ;;
  *) printf 'Layout must be 1b3d, 2b2d, or both.\n' >&2; exit 2 ;;
esac
if (( $# > 2 )) || [[ "$trace" != on && "$trace" != off ]]; then
  printf 'Usage: %s [1b3d|2b2d|both] [on|off]\n' "$0" >&2
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
      scripts/vllm_pd_legacy_trace_{events,overlay}.py; do
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

prepared="${MUSIQUE_PREPARED:-$PD_EXEC_ROOT/data/musique}"
replay="$REPO/scripts/vllm_pd_replay.py"
if [[ -z "${MUSIQUE_PREPARED:-}" && ! -s "$prepared/index/manifest.json" ]]; then
  musique_dev="$("$PYTHON" "$REPO/scripts/vllm_pd_stage_inputs.py" musique "$PD_EXEC_ROOT/data/inputs")"
  "$PYTHON" "$replay" prepare-data --dataset musique --source "$musique_dev" --output "$prepared"
  "$PYTHON" "$replay" build-index --corpus "$prepared/corpus.jsonl" --index-dir "$prepared/index" \
    --embedding-model intfloat/e5-base-v2 --embedding-revision f52bf8ec8c7124536f0efb74aca902b2995e5bcd \
    --embedding-device cuda:0
fi

run_name="musique-legacy-full-blend-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
run_root="$REPO/reports/runs/$run_name"
run_log="$PD_EXEC_ROOT/smoke.log"
if [[ -n "$retained" ]]; then
  run_log="$PD_EXEC_ROOT/$run_name.log"
  git -C "$source_repo" rev-parse HEAD > "$CHECK_ROOT/$run_name-replay-source.txt"
  sha256sum "$lane_patch" > "$CHECK_ROOT/$run_name-lane-support.sha256"
fi
printf 'Run artifacts: %s\n' "$run_root"
"$PYTHON" "$launcher_repo/scripts/vllm_pd_legacy_blend.py" run --topology "$topology" \
  --trace "$trace" --musique-prepared "$prepared" --run-root "$run_root" 2>&1 | tee "$run_log"

report_dir="${EVIDENCESTORE_LEGACY_REPORT_DIR:-$HOME/evidence_store_legacy_reports}"
mkdir -p "$report_dir"
archive="$(mktemp "$report_dir/.${run_name}.XXXXXX")"
tar -czf "$archive" -C "$PD_EXEC_ROOT" check "${run_log##*/}" -C "$REPO/reports/runs" "$run_name"
tar -tzf "$archive" >/dev/null
mv -- "$archive" "$report_dir/$run_name.tar.gz"
sha256sum "$report_dir/$run_name.tar.gz"
printf 'Legacy Full Blend smoke passed. Archive: %s\n' "$report_dir/$run_name.tar.gz"
