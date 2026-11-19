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
activation_receipt="$(mktemp "$node_root/legacy-blend-activation-$SLURM_JOB_ID.XXXXXX")"
trap 'rm -f -- "$activation_receipt"' EXIT
bash "$source_repo/scripts/vllm_pd_legacy_blend_runtime.sh" "$activation_receipt"
source "$(<"$activation_receipt")"

prepared="${MUSIQUE_PREPARED:-$PD_EXEC_ROOT/data/musique}"
replay="$REPO/scripts/vllm_pd_replay.py"
if [[ -z "${MUSIQUE_PREPARED:-}" ]]; then
  musique_dev="$("$PYTHON" "$REPO/scripts/vllm_pd_stage_inputs.py" musique "$PD_EXEC_ROOT/data/inputs")"
  "$PYTHON" "$replay" prepare-data --dataset musique --source "$musique_dev" --output "$prepared"
  "$PYTHON" "$replay" build-index --corpus "$prepared/corpus.jsonl" --index-dir "$prepared/index" \
    --embedding-model intfloat/e5-base-v2 --embedding-revision f52bf8ec8c7124536f0efb74aca902b2995e5bcd \
    --embedding-device cuda:0
fi

run_name="musique-legacy-full-blend-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
run_root="$REPO/reports/runs/$run_name"
printf 'Run artifacts: %s\n' "$run_root"
"$PYTHON" "$REPO/scripts/vllm_pd_legacy_blend.py" run --topology "$topology" \
  --trace "$trace" --musique-prepared "$prepared" --run-root "$run_root" | tee "$PD_EXEC_ROOT/smoke.log"

report_dir="${EVIDENCESTORE_LEGACY_REPORT_DIR:-$HOME/evidence_store_legacy_reports}"
mkdir -p "$report_dir"
archive="$(mktemp "$report_dir/.${run_name}.XXXXXX")"
trap 'rm -f -- "$activation_receipt" "$archive"' EXIT
tar -czf "$archive" -C "$PD_EXEC_ROOT" check smoke.log -C "$REPO/reports/runs" "$run_name"
tar -tzf "$archive" >/dev/null
mv -- "$archive" "$report_dir/$run_name.tar.gz"
sha256sum "$report_dir/$run_name.tar.gz"
printf 'Legacy Full Blend smoke passed. Archive: %s\n' "$report_dir/$run_name.tar.gz"
