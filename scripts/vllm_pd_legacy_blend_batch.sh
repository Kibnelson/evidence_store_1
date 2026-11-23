#!/usr/bin/env bash
# Capture the validated CLAPNQ conversations sequentially with one Full Blend runtime.
set -euo pipefail
umask 077

source_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
plan_only=0
first_four=0
while (( $# )); do
  case "$1" in
    --plan) plan_only=1 ;;
    --first-four) first_four=1 ;;
    *) printf 'Usage: %s [--plan] [--first-four]\n' "$0" >&2; exit 2 ;;
  esac
  shift
done

plan=$(python3 -B "$source_repo/scripts/vllm_pd_legacy_page_campaign.py" plan \
  "$source_repo/workloads/mtrag-multi-turn-repeat.json" "$first_four")
if (( plan_only )); then
  printf '%s\n' "$plan"
  exit
fi

: "${SLURM_JOB_ID:?Run the CLAPNQ capture campaign inside the four-GPU allocation}"
test "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}" = 1
export EVIDENCESTORE_KV_PAGE_CAPTURE_MIB="${EVIDENCESTORE_KV_PAGE_CAPTURE_MIB:-4096}"
if [[ ! "$EVIDENCESTORE_KV_PAGE_CAPTURE_MIB" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Set EVIDENCESTORE_KV_PAGE_CAPTURE_MIB to a positive integer.\n' >&2
  exit 2
fi

node_root="/tmp/$(id -un)"
mkdir -p "$node_root"
[[ -O "$node_root" && ! -L "$node_root" ]]
batch_root="$(mktemp -d "$node_root/mtrag-conversations-$SLURM_JOB_ID.XXXXXX")"
printf '%s\n' "$plan" > "$batch_root/plan.tsv"
printf 'CLAPNQ campaign directory\n  %s\n' "$batch_root"
df -h "$node_root"

activation_receipt=""
trap 'if [[ -n "$activation_receipt" ]]; then rm -f -- "$activation_receipt"; fi' EXIT
if [[ -z "${EVIDENCESTORE_LEGACY_RUNTIME:-}" ]]; then
  activation_receipt="$(mktemp "$batch_root/runtime.XXXXXX")"
  bash "$source_repo/scripts/vllm_pd_legacy_blend_runtime.sh" "$activation_receipt"
  source "$(<"$activation_receipt")"
  export EVIDENCESTORE_LEGACY_RUNTIME="$PD_EXEC_ROOT"
else
  retained="$EVIDENCESTORE_LEGACY_RUNTIME"
  if [[ "$retained" != "$node_root/"* || ! -O "$retained" || -L "$retained" ||
        ! -r "$retained/activate.sh" || ! -s "$retained/check/runtime.json" ]]; then
    printf 'Expected a verified Full Blend runtime under %s.\n' "$node_root" >&2
    exit 1
  fi
  source "$retained/activate.sh"
  [[ "$retained" == "$PD_EXEC_ROOT" ]]
fi

while IFS=$'\t' read -r conversation turns; do
  capture_dir="$batch_root/$conversation"
  mkdir -p "$capture_dir"
  log="$capture_dir/launch.log"
  printf '\nCapturing %s, turns %s\n' "$conversation" "$turns"
  EVIDENCESTORE_LEGACY_REPORT_DIR="$capture_dir" \
    bash "$source_repo/scripts/vllm_pd_legacy_blend_run.sh" \
      1b3d on mtrag-multi-turn-repeat previous-turn-first --trace-kv-pages --capture-kv-pages \
      --page-conversation "$conversation" --page-turns "$turns" 2>&1 | tee "$log"
  receipt=$("$PYTHON" -B "$source_repo/scripts/vllm_pd_legacy_page_campaign.py" record \
    "$REPO" "$capture_dir" "$conversation" "$turns" "$EVIDENCESTORE_KV_PAGE_CAPTURE_MIB")
  printf '%s\n' "$receipt" | tee -a "$batch_root/archives.jsonl"
done < "$batch_root/plan.tsv"

printf '\nVerified campaign archive receipts\n  %s\n' "$batch_root/archives.jsonl"
