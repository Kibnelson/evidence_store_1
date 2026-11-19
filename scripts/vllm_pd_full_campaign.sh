#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

: "${SLURM_JOB_ID:?Run from the active Slurm allocation}"
test "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}" = 1
: "${REPO:?Activate the runtime first}"
: "${PYTHON:?Activate the runtime first}"
: "${RUN_ROOT:?Run this from the shell that completed the smoke}"
: "${MODEL:?Run this from the shell that completed the smoke}"
: "${MODEL_REVISION:?Run this from the shell that completed the smoke}"

cd "$REPO"
trap 'printf "Full campaign stopped at line %s. Artifacts: %s\n" "$LINENO" "$RUN_ROOT" >&2' ERR
smoke_root="${SMOKE_ROOT:-$RUN_ROOT/smoke}"
smoke_topology="${SMOKE_TOPOLOGY:-1p3d}"
"$PYTHON" - "$smoke_root" "$smoke_topology" <<'PYTHON'
import json
from pathlib import Path
import sys

root, topology = Path(sys.argv[1]), sys.argv[2]
for mode in ("native", "shared"):
    receipt = root / f"{topology}-{mode}" / "smoke-check.json"
    if not receipt.is_file() or json.loads(receipt.read_text())["status"] != "passed":
        raise SystemExit(f"Full campaign needs a passed smoke receipt at {receipt}")
PYTHON
IFS=@ read -r E5_MODEL E5_REVISION < "$RUN_ROOT/e5-model.txt"
: "${E5_MODEL:?Smoke embedding model is missing}"
: "${E5_REVISION:?Smoke embedding revision is missing}"
helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPLAY="$helper_dir/vllm_pd_replay.py"
DATASET_REPORT="$helper_dir/vllm_pd_dataset_report.py"
DATASET_PUBLISH="$helper_dir/vllm_pd_publish_dataset.py"
input_dir="$RUN_ROOT/inputs"
printf 'Starting full campaign for Slurm job %s in %s\n' "$SLURM_JOB_ID" "$RUN_ROOT"
df -h "$RUN_ROOT"
git -C "$helper_dir/.." rev-parse HEAD > "$RUN_ROOT/full-campaign-source.txt"
MUSIQUE_DEV="${MUSIQUE_DEV:-$("$PYTHON" "$helper_dir/vllm_pd_stage_inputs.py" musique "$input_dir")}"
TWOWIKI_DEV="${TWOWIKI_DEV:-$("$PYTHON" "$helper_dir/vllm_pd_stage_inputs.py" 2wiki "$input_dir")}"

sha256sum "$MUSIQUE_DEV" "$TWOWIKI_DEV" > "$RUN_ROOT/input-sha256.txt"

prepare_qa() {
  local dataset="$1" source="$2" output="$RUN_ROOT/data/$1"
  if [[ ! -f "$output/manifest.json" ]]; then
    "$PYTHON" "$REPLAY" prepare-data --dataset "$dataset" --source "$source" --output "$output"
  fi
}
prepare_qa musique "$MUSIQUE_DEV"
prepare_qa 2wiki "$TWOWIKI_DEV"
"$PYTHON" - "$RUN_ROOT" "$MUSIQUE_DEV" "$TWOWIKI_DEV" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
sources = dict(zip(("musique", "2wiki"), map(Path, sys.argv[2:]), strict=True))
expected = {"musique": (2417, 0), "2wiki": (12576, 0), "mtrag": (842, 110),
            "mtrag/clapnq": (224, 29), "mtrag/cloud": (205, 26),
            "mtrag/fiqa": (199, 27), "mtrag/govt": (214, 28)}
workloads = {}
for name in ("musique", "2wiki", "mtrag"):
    manifest = json.loads((root / "data" / name / "manifest.json").read_text())
    if name in sources:
        with sources[name].open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != manifest["source_sha256"]:
                raise SystemExit(f"{name}: prepared data came from a different input file")
    workloads[name] = manifest
    workloads.update({f"{name}/{domain}": row for domain, row in manifest.get("domains", {}).items()})
observed = {name: (row["query_rows"], row["multi_turn_conversations"]) for name, row in workloads.items()}
receipt = {name: {"requests": count, "multi_turn_conversations": conversations}
           for name, (count, conversations) in observed.items()}
(root / "dataset-counts.json").write_text(json.dumps(receipt, indent=2) + "\n")
if observed != expected:
    raise SystemExit(f"Dataset counts differ from the planned splits. Inspect {root / 'dataset-counts.json'}")
PY

DATASETS=(musique 2wiki mtrag/clapnq mtrag/cloud mtrag/fiqa mtrag/govt)
for dataset in "${DATASETS[@]}"; do
  dataset_dir="$RUN_ROOT/data/$dataset"
  printf 'Building HNSW index for %s\n' "$dataset"
  "$PYTHON" "$REPLAY" build-index --corpus "$dataset_dir/corpus.jsonl" \
    --index-dir "$dataset_dir/index" --embedding-model "$E5_MODEL" \
    --embedding-revision "$E5_REVISION" --embedding-device cuda:0
done

PD_CONCURRENCY="${PD_CONCURRENCY:-8}"
export PYTHON REPO REPLAY RUN_ROOT MODEL MODEL_REVISION PD_CONCURRENCY
export DATASET_REPORT DATASET_PUBLISH

cat > "$RUN_ROOT/workload.sh" <<'WORKLOAD'
#!/usr/bin/env bash
set -euo pipefail
cohort_topology="$1"
cohort_mode="$2"
run_dir="$EVIDENCESTORE_RUN_DIR"
gateway="$EVIDENCESTORE_GATEWAY"
run_replay() {
  "$PYTHON" "$REPLAY" replay --gateway "$gateway" --run-dir "$run_dir" \
    --model "$MODEL" "$@"
}

run_replay --label controlled --frozen "$RUN_ROOT/data/controlled-$cohort_topology.jsonl"
"$PYTHON" "$REPO/scripts/vllm_pd_report.py" --run-dir "$run_dir" --capture after
"$PYTHON" "$REPO/scripts/vllm_pd_report.py" --run-dir "$run_dir" --check --all-pairs
for dataset in musique 2wiki mtrag/clapnq mtrag/cloud mtrag/fiqa mtrag/govt; do
  label="${dataset//\//-}"
  dataset_dir="$RUN_ROOT/data/$dataset"
  printf 'Replaying %s for %s / %s\n' "$dataset" "$cohort_topology" "$cohort_mode"
  "$PYTHON" "$DATASET_REPORT" --run-dir "$run_dir" --dataset "$dataset" --begin
  if [[ "$cohort_topology/$cohort_mode" == 1p3d/native ]]; then
    run_replay --label "$label" --concurrency "$PD_CONCURRENCY" --queries "$dataset_dir/queries.jsonl" \
      --index-dir "$dataset_dir/index" --tokenizer "$MODEL" --tokenizer-revision "$MODEL_REVISION"
  else
    run_replay --label "$label" --concurrency "$PD_CONCURRENCY" \
      --frozen "$RUN_ROOT/1p3d-native/client/$label/frozen.jsonl"
    cmp -- "$RUN_ROOT/1p3d-native/client/$label/frozen.jsonl" "$run_dir/client/$label/frozen.jsonl"
  fi
  "$PYTHON" - "$RUN_ROOT/dataset-counts.json" "$run_dir/client/$label/manifest.json" "$dataset" <<'PY'
import json
from pathlib import Path
import sys

counts_path, replay_path, dataset = sys.argv[1:]
expected = json.loads(Path(counts_path).read_text())[dataset]["requests"]
receipt = json.loads(Path(replay_path).read_text())
if receipt["requested"] != expected or receipt["succeeded"] != expected:
    raise SystemExit(f"{dataset}: expected {expected} completed requests; inspect {replay_path}")
PY
  "$PYTHON" "$DATASET_REPORT" --run-dir "$run_dir" --dataset "$dataset" --finish
  dataset_summary="$run_dir/datasets/$label/summary.json"
  if ! "$PYTHON" "$DATASET_PUBLISH" --report "$dataset_summary"; then
    touch "$dataset_summary.pending"
    printf 'Git publication pending for %s. Dataset report: %s\n' "$dataset" "$dataset_summary" >&2
  fi
done
WORKLOAD

for topology in 1p3d 2p2d; do
  "$PYTHON" "$REPLAY" fixture --topology "$topology" --tokenizer "$MODEL" \
    --tokenizer-revision "$MODEL_REVISION" --evidence-repeats 12 \
    --output "$RUN_ROOT/data/controlled-$topology.jsonl"
  for mode in native shared; do
    run_dir="$RUN_ROOT/$topology-$mode"
    launch=("$PYTHON" scripts/vllm_pd_cluster.py --run-dir "$run_dir" --topology "$topology"
      --trace on --model "$MODEL" --revision "$MODEL_REVISION")
    if [[ "$mode" == shared ]]; then
      launch+=(--lmcache --lmcache-on-decode --prefix-cache on
        --lmcache-size-gb "${PD_LMCACHE_SIZE_GB:-8}" --lmcache-chunk-size 256)
    else
      launch+=(--prefix-cache off)
    fi
    printf 'Starting cohort %s / %s\n' "$topology" "$mode"
    "${launch[@]}" -- bash "$RUN_ROOT/workload.sh" "$topology" "$mode"
    "$PYTHON" scripts/vllm_pd_report.py --run-dir "$run_dir" --check --all-pairs
  done
done

printf 'Completed all four dataset cohorts. Artifacts: %s\n' "$RUN_ROOT"
for pending in "$RUN_ROOT"/{1p3d,2p2d}-{native,shared}/datasets/*/summary.json.pending; do
  [[ -f "$pending" ]] || continue
  if "$PYTHON" "$DATASET_PUBLISH" --report "${pending%.pending}"; then
    rm -- "$pending"
  else
    printf 'Git publication still pending. Dataset report: %s\n' "${pending%.pending}" >&2
  fi
done
