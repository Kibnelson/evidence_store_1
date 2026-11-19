#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

if [[ "${1:-}" == --help && $# == 1 ]]; then
  cat <<'USAGE'
Run inside the four-GPU Slurm allocation after sourcing the verified runtime's activate.sh.
Run without arguments for the smoke or pass --full for one live pass over MuSiQue in each cohort.
Set MUSIQUE_PREPARED to the retained full prepared MuSiQue directory or
set MUSIQUE_DEV to the staged pinned MuSiQue-Ans development JSONL.
Set MUSIQUE_INDEX to reuse an index saved outside the prepared directory.
The runner reuses a matching retained index or builds the full MuSiQue index.
USAGE
  exit 0
fi
run_kind=smoke
if [[ "${1:-}" == --full ]]; then
  run_kind=full
  shift
fi
if (( $# )); then
  printf 'Run without arguments, with --full, or with --help.\n' >&2
  exit 2
fi

: "${SLURM_JOB_ID:?Run this inside the four-GPU Slurm allocation}"
test "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}" = 1
: "${REPO:?Source the runtime activation script first}"
: "${PYTHON:?Source the runtime activation script first}"
if [[ -z "${MUSIQUE_PREPARED:-}" && -z "${MUSIQUE_DEV:-}" ]]; then
  printf 'Set MUSIQUE_PREPARED or set MUSIQUE_DEV to the JSONL from scripts/vllm_pd_stage_inputs.py musique INPUT_DIR.\n' >&2
  exit 2
fi

cd "$REPO"
helpers=(vllm_pd_chunk_smoke_inputs.py vllm_pd_chunk_report.py)
if [[ "$run_kind" == full ]]; then
  helpers+=(vllm_pd_musique_repeat_report.py)
fi
for helper in "${helpers[@]}"; do
  if [[ ! -r "scripts/$helper" ]]; then
    printf 'The active runtime is missing scripts/%s. Activate a runtime with the chunk tracing source.\n' "$helper" >&2
    exit 2
  fi
done

model="Qwen/Qwen3-4B-Instruct-2507"
revision="cdbee75f17c01a7cc42f958dc650907174af0554"
embedder="intfloat/e5-base-v2"
embedding_revision="f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
replay="$REPO/scripts/vllm_pd_replay.py"
dataset_report="$REPO/scripts/vllm_pd_dataset_report.py"
chunk_report="$REPO/scripts/vllm_pd_chunk_report.py"
smoke_inputs="$REPO/scripts/vllm_pd_chunk_smoke_inputs.py"

mkdir -p "$REPO/reports/runs"
run_root="$(mktemp -d "$REPO/reports/runs/musique-chunk-${run_kind}-${SLURM_JOB_ID}.XXXXXX")"
trap 'printf "MuSiQue chunk %s stopped at line %s. Artifacts: %s\n" "$run_kind" "$LINENO" "$run_root" >&2' ERR
printf 'MuSiQue chunk %s artifacts: %s\n' "$run_kind" "$run_root"

prepared="${MUSIQUE_PREPARED:-$run_root/musique}"
if [[ -z "${MUSIQUE_PREPARED:-}" ]]; then
  "$PYTHON" "$replay" prepare-data --dataset musique --source "$MUSIQUE_DEV" --output "$prepared"
fi
inputs="$run_root/inputs"
prepare=("$PYTHON" "$smoke_inputs" prepare --prepared "$prepared" --output "$inputs"
  --model "$model" --revision "$revision" --embedding-model "$embedder" --embedding-revision "$embedding_revision")
if [[ "$run_kind" == full ]]; then
  prepare+=(--full)
fi
index="${MUSIQUE_INDEX:-$prepared/index}"
if [[ -r "$index/manifest.json" ]]; then
  "${prepare[@]}" --retained-index "$index"
else
  if [[ -n "${MUSIQUE_INDEX:-}" ]]; then
    printf 'Missing index manifest at %s/manifest.json\n' "$index" >&2
    exit 2
  fi
  "${prepare[@]}"
  index="$run_root/index"
  "$PYTHON" "$replay" build-index --corpus "$prepared/corpus.jsonl" --index-dir "$index" \
    --embedding-model "$embedder" --embedding-revision "$embedding_revision" --embedding-device cuda:0
fi

for mode in native shared; do
  run_dir="$run_root/2p2d-$mode"
  launch=("$PYTHON" scripts/vllm_pd_cluster.py --run-dir "$run_dir" --topology 2p2d
    --model "$model" --revision "$revision" --trace on)
  if [[ "$mode" == shared ]]; then
    launch+=(--prefix-cache on --lmcache --lmcache-on-decode --trace-lmcache-transfers
      --lmcache-size-gb "${PD_LMCACHE_SIZE_GB:-8}" --lmcache-chunk-size 256)
  else
    launch+=(--prefix-cache off)
  fi
  printf 'Starting MuSiQue %s / 2p2d-%s\n' "$run_kind" "$mode"
  EVIDENCESTORE_TRACE_GPU=1 "${launch[@]}" -- bash -c '
    set -euo pipefail
    python="$1" replay="$2" dataset_report="$3" model="$4" revision="$5" inputs="$6" index="$7"
    mode="$8" smoke_inputs="$9" run_kind="${10}"
    run_dir="$EVIDENCESTORE_RUN_DIR"
    send=("$python" "$replay" replay --gateway "$EVIDENCESTORE_GATEWAY" --run-dir "$run_dir"
      --model "$model" --concurrency 1)
    drain_cache() {
      if [[ "$mode" == shared ]]; then
        "$python" "$smoke_inputs" drain --run-dir "$run_dir"
      fi
    }
    if [[ "$run_kind" == smoke ]]; then
      "${send[@]}" --label chunk-probes --frozen "$inputs/probes.jsonl"
      drain_cache
    fi
    "$python" "$dataset_report" --run-dir "$run_dir" --dataset musique --begin
    "${send[@]}" --label musique --queries "$inputs/queries.jsonl" --index-dir "$index" \
      --tokenizer "$model" --tokenizer-revision "$revision" --trace-chunk-stages
    drain_cache
    "$python" "$dataset_report" --run-dir "$run_dir" --dataset musique --finish --timeline-limit 7
  ' musique-replay "$PYTHON" "$replay" "$dataset_report" "$model" "$revision" "$inputs" "$index" "$mode" "$smoke_inputs" "$run_kind"
  report=("$PYTHON" "$chunk_report" --run-dir "$run_dir" --label musique)
  if [[ "$run_kind" == smoke ]]; then
    report+=(--label chunk-probes --check-smoke)
  fi
  "${report[@]}"
done

if [[ "$run_kind" == smoke ]]; then
  "$PYTHON" "$smoke_inputs" verify-pair --smoke-root "$run_root"
else
  "$PYTHON" "$REPO/scripts/vllm_pd_musique_repeat_report.py" --run-root "$run_root" --queries "$inputs/queries.jsonl"
fi
printf 'MuSiQue chunk %s passed. Reports are under %s/2p2d-{native,shared}/chunk-review/.\n' "$run_kind" "$run_root"
