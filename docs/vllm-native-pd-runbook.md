# Native vLLM PD runbook

This runbook compares a full native NIXL handoff with **one shared LMCache MP server for P and D** and native decode prefix reuse enabled. Both layouts use vLLM's native NIXL pull. A cold D pulls the full prompt KV. A warm D can use blocks already resident in its GPU cache and pull the missing part. The LMCache runs use the public prefix cache connector. [LMCache multiprocess deployment](https://docs.lmcache.ai/mp/deployment.html)

The shared arm configures NIXL first and LMCache second on both roles. Native `MultiConnector` selects the first connector that reports a positive external match. The trace records the selected connector, D's local prefix, and completed NIXL bytes. This shows whether a request actually used LMCache for D. The manual PD recipe uses dedicated servers and the operator documents one per node. Verify the shared single-node configuration with the controlled smoke before running the datasets. [Pinned selector](https://github.com/vllm-project/vllm/blob/b24eda7127a9c5c317f699db3f580135eec7322b/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py) · [LMCache PD operator](https://docs.lmcache.ai/mp/operator.html#pd-disaggregation) · [Manual PD recipe](https://github.com/LMCache/LMCache/blob/042d47abdef0d460cf2cb0d5720e8a59280cda8a/docs/source/mp/disaggregated_prefill.rst)

The default is **1P/3D**. The **2P/2D** arm reassigns one GPU from decode to prefill. Each worker uses one GPU with TP=1. The gateway sends a one-token request to P, receives the native KV coordinates, selects D, and relays D's completion stream. Force all P→D pairs in the controlled tests. Use the gateway's round-robin routing for the datasets.

Run section 0 on the cluster login node. Run section 1 onward from the compute-node shell of **one single-node Slurm allocation with four GPUs**. The launcher supervises its workers, gateway, and workload inside that allocation. Startup, both layouts, dataset ingestion and replay, shutdown, and reporting all use the same allocation. The earlier run card supplies the model, runtime settings, and frozen input format. [Prior native PD run card](../../context_pilot_paper/docs/evidencestore/implementation/scheduler_v2_phase2_native_and_three_modes_run_card.md)

For a smoke-only run, follow [login and allocation](#0-prepare-the-login-shell-and-request-the-allocation), [runtime setup](#1-prepare-the-runtime-and-the-inputs), and [FiQA smoke](#one-dataset-smoke). After both smoke receipts pass, you can [save the receipts](#3-check-the-receipts-in-the-allocation) or continue to the [full dataset inputs](#dataset-inputs).

## What carries over from the earlier run

| Earlier native run | This run |
| --- | --- |
| `gpu-grad-01`, one node, four GPUs, 16 CPUs, 192 GiB and a 12-hour limit. The card submitted measurement and review separately. | Use one compute-node shell with the same allocation shape. The launcher starts and cleans up each cohort within it. |
| Qwen 3 4B pinned revision, BF16, TP=1, 16-token blocks, 32 sequences, 90% GPU allocation, eager and synchronous scheduling. The five-conversation test used 9,216 tokens for both context and batch limits. | Match those settings in the historical check. The controlled and dataset cohorts use the default 10,240-token limit. |
| The gateway chose D after P advertised native KV. Native prefix caching was off. The old runner and NIXL 1.4.1 were pinned. | The historical check keeps that routing and full handoff. The shared arm enables prefix caching and adds one LMCache server for P and D. This fork uses the current runner and NIXL 1.5.2. Compare input and route behavior against the old run and measure performance on the new build. |
| The card retained frozen prompts, separate warm-up and measured requests, and raw metrics after counter reconciliation. | Keep the original input and phase receipts. Save raw native and LMCache metrics, reconcile with NIXL completion traces, and check cold-to-warm output token parity in the shared smoke. |

## 0. Prepare the login shell and request the allocation

**Retry after a quota failure.** A line such as `srun: ... task 0: Exited` followed by the login prompt means the previous allocation has ended. From that login shell, update the checkout and request a fresh allocation below. If you are still in a four-GPU allocation and your checkout is current, start at [section 1](#1-prepare-the-runtime-and-the-inputs). The build helper creates a fresh scratch venv for each attempt.

### 0.1 Check Slurm and update the checkout

Use the cluster login shell where `srun --version` worked. To open a research login, replace `YOUR_CS_USERNAME` with your CS account. [McGill research Slurm instructions](https://docs.sci.mcgill.ca/COMP/slurm/#research-usage)

```bash
ssh YOUR_CS_USERNAME@research.cs.mcgill.ca
```

From the SSH shell outside `tmux`, load and verify Slurm as in the Phase 2 launcher. The version check succeeded on `teach-node-09`. [Phase 2 launcher](../../context_pilot_paper/scripts/scheduler_v2_phase1_native_1p3d_gpu_card.sh#L29)

```bash
module load slurm && srun --version
```

On a first visit, clone into a directory shared with the compute node.

```bash
git clone https://github.com/Kibnelson/evidence_store_1.git \
  "${EVIDENCESTORE_CLUSTER_REPO:-$HOME/evidence_store_1}"
```

For an existing clone, pull `main`. If your checkout lives elsewhere, export `EVIDENCESTORE_CLUSTER_REPO` with its absolute path in the Bash shell you use to submit the allocation.

```bash
git -C "${EVIDENCESTORE_CLUSTER_REPO:-$HOME/evidence_store_1}" pull --ff-only origin main &&
  git -C "${EVIDENCESTORE_CLUSTER_REPO:-$HOME/evidence_store_1}" rev-parse --short HEAD
```

### 0.2 Keep the login shell in tmux (optional)

For a direct run, continue to section 0.3 from the verified Bash login shell. For `tmux`, start the dedicated server from the same SSH shell after checking Slurm. Its first pane starts Bash with the inherited Slurm environment. [Earlier tmux setup](../../context_pilot_paper/docs/evidencestore/implementation/scheduler_v1_native_pd_controls_runbook.md#40-login-host-enter-a-persistent-bash-shell) · [tmux environment and socket options](https://man.openbsd.org/tmux.1#GLOBAL_AND_SESSION_ENVIRONMENT)

```bash
tmux -L evidence-store-slurm -f /dev/null new-session -s evidence-store-pd \
  /bin/bash --noprofile --norc -i
```

At the new pane's Bash prompt, verify it inherited Slurm. If this fails, continue in the original SSH shell where the check passed.

```bash
srun --version
```

Detach with `Ctrl+b` then `d`. To resume an existing session, reconnect to the same login node and attach on the same socket.

```bash
tmux -L evidence-store-slurm attach-session -t evidence-store-pd
```

### 0.3 Request the four-GPU allocation

Paste the complete block into the verified login Bash shell. Once the `gpu-grad-01` prompt appears, run section 1 there.

```bash
(
  login_host=$(hostname -s)
  printf 'Login host: %s\n' "$login_host"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    printf 'Already inside Slurm allocation %s. Use section 1 if this is your four-GPU job.\n' "$SLURM_JOB_ID" >&2
    exit 1
  fi
  cd "${EVIDENCESTORE_CLUSTER_REPO:-$HOME/evidence_store_1}" || exit 1
  for required in scripts/vllm_pd_runtime.sh scripts/vllm_pd_cluster.py \
    third_party/vllm-native-pd.bundle third_party/vllm-fork.json; do
    if [[ ! -r "$required" || ! -s "$required" ]]; then
      printf 'Missing, unreadable, or empty run input: %s/%s\n' "$PWD" "$required" >&2
      exit 1
    fi
  done
  if ! command -v srun >/dev/null; then
    if ! command -v module >/dev/null || ! module load slurm; then
      printf 'Could not load Slurm on %s. Run the Slurm check in the SSH shell outside tmux.\n' "$login_host" >&2
      exit 1
    fi
  fi
  if ! command -v srun || ! srun --version; then
    printf 'The Slurm client srun is unavailable on %s after loading the module.\n' "$login_host" >&2
    exit 1
  fi
  srun -p all -A oana2026 -q research12h -N 1 -w gpu-grad-01 \
    --gpus=4 -c 16 --mem=192G -t 12:00:00 --pty bash -l
)
```

## 1. Prepare the runtime and the inputs

If setup failed and the `gpu-grad-01` prompt is still open with a Slurm job ID, update the shared checkout from that shell and retry section 1.2 in the same allocation. The previous placeholder value for `CUDA_HOME` can be cleared before retrying.

```bash
git -C "${EVIDENCESTORE_CLUSTER_REPO:-$HOME/evidence_store_1}" pull --ff-only origin main
unset CUDA_HOME
```

### 1.1 Verify the compute shell and toolkit

Expect `gpu-grad-01`, a job ID, and four RTX A5000 GPUs. The version below describes the host toolkit. CUDA 12.6 on the host is expected and setup installs the matching CUDA 13.2 compiler on scratch.

```bash
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf 'Host %s, Slurm job %s\n' "$(hostname -s)" "$SLURM_JOB_ID"
  nvidia-smi --query-gpu=uuid,name,driver_version --format=csv
  if command -v nvcc >/dev/null; then
    nvcc --version
  fi
else
  printf 'Request the allocation in section 0.3 first.\n' >&2
fi
```

### 1.2 Build and activate on node-local scratch

The helper stages the committed experiment tree, venv, editable vLLM and LMCache trees, and run output under `/tmp/$USER` on the GPU node. It reuses node-local uv and Hugging Face caches and prints free disk space and inodes before installation. [Prior allocation-local setup](../../context_pilot_paper/docs/evidencestore/implementation/scheduler_v2_phase1_comparison_gpu_steps.sh#L96) · [uv storage paths](https://docs.astral.sh/uv/reference/storage/)

The helper uses system CUDA 13.2 when available. Otherwise it installs NVIDIA's `cuda-toolkit[nvcc,cccl]==13.2.1` package into the scratch venv. This shares the CUDA runtime pinned by Torch. The helper selects the resulting `CUDA_HOME` and checks that it can compile and link a small RTX A5000 program. [NVIDIA CUDA toolkit package](https://pypi.org/project/cuda-toolkit/13.2.1/) · [Pinned Torch wheel metadata](https://download-r2.pytorch.org/whl/cu132/torch-2.14.0%2Bcu132-cp312-cp312-manylinux_2_28_x86_64.whl.metadata)

For Python-only vLLM edits, the helper installs the instrumented source in editable mode and uses the exact upstream wheel for native libraries. The full-source path uses GCC/G++ 13 or later. Use `CC` and `CXX` to select a compatible compiler pair when needed. [Previous native install](../../context_pilot_paper/scripts/scheduler_v1_prepare_rtx_host_runtime.sh#L248) · [Pinned vLLM build instructions](https://github.com/vllm-project/vllm/blob/b24eda7127a9c5c317f699db3f580135eec7322b/docs/getting_started/installation/gpu.cuda.inc.md#python-only-build) · [Published CUDA 13.2 wheel metadata](https://wheels.vllm.ai/b24eda7127a9c5c317f699db3f580135eec7322b/cu132/vllm/metadata.json)

```bash
if cd "${EVIDENCESTORE_CLUSTER_REPO:-$HOME/evidence_store_1}" &&
    bash scripts/vllm_pd_runtime.sh; then
  printf 'Paste the source command printed above into this allocation shell.\n'
else
  printf 'Setup failed. Inspect check/setup.log under the printed runtime scratch directory before retrying.\n' >&2
fi
```

Setup succeeds when it prints `Runtime verified` and a `source /tmp/.../activate.sh` command. Run that exact `source` command in the same allocation shell. It activates the venv and changes into the scratch copy. A failed setup leaves the shell available to inspect its log and retry section 1.2.

Verify the setup receipt and cache path before starting the smoke. The cache should be `/tmp/$USER/uv-cache`. LMCache builds against the same Torch installation as vLLM. [Pinned LMCache installation from source](https://github.com/LMCache/LMCache/blob/042d47abdef0d460cf2cb0d5720e8a59280cda8a/docs/source/getting_started/installation.rst#L283)

```bash
pwd
cat "$CHECK_ROOT/cuda-toolkit.txt"
cat "$CHECK_ROOT/runtime.json"
uv cache dir
```

### One-dataset smoke

Run this in the same allocation before building the full indexes. The default layout is `1p3d`. Set `SMOKE_TOPOLOGY=2p2d` to smoke-test that layout instead. Each launch runs the controlled probes on every P→D pair, then replays the same 10 real MT-RAG Human FiQA requests through live HNSW. The sample includes the two shortest complete conversations (four turns each) and the opening turn from two other conversations. Each conversation runs in turn order with fixed dataset reference history.

The smoke index has 83 passages from FiQA, including the 19 reference passages used by the sample and 64 other passages. The full campaign builds indexes from the complete corpora. The smoke needs only the pinned MT-RAG checkout and runtime models. Stage the MuSiQue and 2Wiki files for the full campaign.

```bash
RUN_ROOT="$REPO/reports/runs/native-vllm-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
SMOKE_ROOT="$RUN_ROOT/smoke"
SMOKE_TOPOLOGY="${SMOKE_TOPOLOGY:-1p3d}"
mkdir -p "$SMOKE_ROOT"
MODEL="Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION="cdbee75f17c01a7cc42f958dc650907174af0554"
E5_MODEL="intfloat/e5-base-v2"
E5_REVISION="${E5_REVISION:-$("$PYTHON" -c 'from huggingface_hub import model_info; print(model_info("intfloat/e5-base-v2").sha)')}"
printf '%s\n' "$E5_MODEL@$E5_REVISION" > "$RUN_ROOT/e5-model.txt"
uv pip freeze > "$RUN_ROOT/python-packages.txt"
REPLAY="$REPO/scripts/vllm_pd_replay.py"
"$PYTHON" "$REPLAY" prepare-data --dataset mtrag --source "$REPO/third_party/checkouts/mt-rag" \
  --output "$RUN_ROOT/data/mtrag"
sample="$SMOKE_ROOT/fiqa-sample"
fiqa="$RUN_ROOT/data/mtrag/fiqa"
"$PYTHON" "$REPLAY" smoke-sample --queries "$fiqa/queries.jsonl" --corpus "$fiqa/corpus.jsonl" \
  --output "$sample"
"$PYTHON" "$REPLAY" build-index --corpus "$sample/corpus.jsonl" --index-dir "$sample/index" \
  --embedding-model "$E5_MODEL" --embedding-revision "$E5_REVISION" --embedding-device cuda:0
fixture="$SMOKE_ROOT/controlled-$SMOKE_TOPOLOGY.jsonl"
"$PYTHON" "$REPLAY" fixture --topology "$SMOKE_TOPOLOGY" --tokenizer "$MODEL" \
  --tokenizer-revision "$MODEL_REVISION" --evidence-repeats 12 --output "$fixture"

for mode in native shared; do
  run_dir="$SMOKE_ROOT/$SMOKE_TOPOLOGY-$mode"
  launch=("$PYTHON" scripts/vllm_pd_cluster.py --run-dir "$run_dir" --topology "$SMOKE_TOPOLOGY"
    --model "$MODEL" --revision "$MODEL_REVISION" --trace on)
  if [[ "$mode" == shared ]]; then
    launch+=(--prefix-cache on --lmcache --lmcache-on-decode
      --lmcache-size-gb "${PD_LMCACHE_SIZE_GB:-8}" --lmcache-chunk-size 256)
  else
    launch+=(--prefix-cache off)
  fi
  EVIDENCESTORE_TRACE_GPU=1 "${launch[@]}" -- bash -c '
    set -euo pipefail
    python="$1" replay="$2" model="$3" revision="$4" fixture="$5" sample="$6"
    send=("$python" "$replay" replay --gateway "$EVIDENCESTORE_GATEWAY"
      --run-dir "$EVIDENCESTORE_RUN_DIR" --model "$model")
    "${send[@]}" --label controlled --frozen "$fixture"
    "${send[@]}" --label mtrag-smoke --queries "$sample/queries.jsonl" --index-dir "$sample/index" \
      --tokenizer "$model" --tokenizer-revision "$revision" --concurrency 2
  ' smoke "$PYTHON" "$REPLAY" "$MODEL" "$MODEL_REVISION" "$fixture" "$sample"
  "$PYTHON" scripts/vllm_pd_report.py --run-dir "$run_dir" --check --all-pairs \
    --timeline-limit 64 --smoke-sample "$sample"
done
```

Both runs must write `smoke-check.json` before proceeding. The checks cover completed turns, retrieval stage order, doc and chunk IDs and token spans across the gateway, P and D execution, NIXL payload reconciliation, and streaming through the last token. The shared arm uses the same D for cold and exact-repeat probes to verify that warm D-local reuse reduces NIXL bytes. Open each `timeline.json` to inspect the real requests. The receipt lists their request IDs.

```bash
for mode in native shared; do
  receipt="$SMOKE_ROOT/$SMOKE_TOPOLOGY-$mode/smoke-check.json"
  if [[ -s "$receipt" ]]; then
    printf '%s smoke passed: %s\n' "$mode" "$receipt"
  else
    printf 'Missing %s smoke receipt: %s\n' "$mode" "$receipt" >&2
  fi
done
```

For a smoke-only run, save the results in section 3 and exit the allocation. Continue below to run the full datasets in the same job.

### Dataset inputs

Set the following input paths and run the preparation commands in the same shell. `MUSIQUE_DEV` points to the labeled MuSiQue-Ans `musique_ans_v1.0_dev.jsonl`. `TWOWIKI_DEV` points to the labeled 2WikiMultiHopQA `dev.json` from the April 7 archive with corrected sentence segmentation. The dataset campaign consumes the full selected development query files and all four MT-RAG domains. [MuSiQue data](https://github.com/StonyBrookNLP/musique) · [2WikiMultiHopQA April 7 archive](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip) · [MT-RAG data](https://github.com/IBM/mt-rag-benchmark)

| Evaluation input | Multi-turn conversations | Requests per full cohort |
| --- | ---: | ---: |
| MuSiQue-Ans v1.0 dev | Single-turn questions | 2,417 |
| 2WikiMultiHopQA April 7 dev | Single-turn questions | 12,576 |
| MT-RAG Human, ClapNQ | 29 | 224 |
| MT-RAG Human, Cloud | 26 | 205 |
| MT-RAG Human, FiQA | 27 | 199 |
| MT-RAG Human, Government | 28 | 214 |
| **Total** | **110 multi-turn conversations** | **15,835** |

The two QA counts come from their published dev splits. The MT-RAG counts come from unique conversation and task IDs in the pinned Human generation file. Four full cohorts give **63,340 dataset requests**, plus the controlled probes and historical check. This is the expected workload. Preparation checks the supplied files before indexing, and replay receipts verify how many completed. [MuSiQue-Ans Table 2](https://aclanthology.org/2022.tacl-1.31.pdf#page=8) · [2WikiMultiHopQA Table 1](https://aclanthology.org/2020.coling-main.580.pdf#page=6) · [Pinned MT-RAG Human tasks](https://github.com/IBM/mt-rag-benchmark/blob/e0970de1b1207c2a9506adf238e6e1a22037e38a/mtrag-human/generation_tasks/reference.jsonl)

```bash
test -s "$SMOKE_ROOT/$SMOKE_TOPOLOGY-native/smoke-check.json"
test -s "$SMOKE_ROOT/$SMOKE_TOPOLOGY-shared/smoke-check.json"
: "${MUSIQUE_DEV:?Set the path to musique_ans_v1.0_dev.jsonl}"
: "${TWOWIKI_DEV:?Set the path to the labeled 2WikiMultiHopQA dev.json}"
sha256sum "$MUSIQUE_DEV" "$TWOWIKI_DEV" > "$RUN_ROOT/input-sha256.txt"

"$PYTHON" "$REPLAY" prepare-data --dataset musique --source "$MUSIQUE_DEV" \
  --output "$RUN_ROOT/data/musique"
"$PYTHON" "$REPLAY" prepare-data --dataset 2wiki --source "$TWOWIKI_DEV" \
  --output "$RUN_ROOT/data/2wiki"
"$PYTHON" - "$RUN_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = {"musique": (2417, 0), "2wiki": (12576, 0), "mtrag": (842, 110),
            "mtrag/clapnq": (224, 29), "mtrag/cloud": (205, 26),
            "mtrag/fiqa": (199, 27), "mtrag/govt": (214, 28)}
workloads = {}
for name in ("musique", "2wiki", "mtrag"):
    manifest = json.loads((root / "data" / name / "manifest.json").read_text())
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
  "$PYTHON" "$REPLAY" build-index --corpus "$dataset_dir/corpus.jsonl" \
    --index-dir "$dataset_dir/index" --embedding-model "$E5_MODEL" \
    --embedding-revision "$E5_REVISION" --embedding-device cuda:0
done
```

The indexes use the allocated GPU for corpus embedding before the serving workers start. The live retrieval run embeds queries on the CPU, records search stages, and freezes the ranked IDs and resulting tokenized prompts. Paired runs read those same prompt IDs. `dispatch_to_first_token_ms` measures serving after dispatch. For a throughput or latency claim, run each arm with the same frozen input and concurrency so they share a comparable CPU and arrival load. The live capture also reports latency from before embedding. Set `E5_REVISION` to the local model's exact revision when running with a populated model cache and no Hub access.

## 2. Run both layouts and all datasets

The launcher starts four independent workers, validates the physical GPUs and editable fork, captures native counters before and after the workload, and tears down the process groups. `--dry-run` prints the effective settings. The native full-handoff reference runs with `--prefix-cache off`. The shared arm runs with `--lmcache --lmcache-on-decode --prefix-cache on`. Each cohort starts with fresh worker and server caches. To measure the prefill-only LMCache control in isolation, use `--lmcache --prefix-cache off`.

Run the workload from the same shell that prepared the inputs.

```bash
PD_CONCURRENCY="${PD_CONCURRENCY:-8}"
export PYTHON REPO REPLAY RUN_ROOT MODEL MODEL_REVISION PD_CONCURRENCY

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
  if [[ "$cohort_topology/$cohort_mode" == 1p3d/native ]]; then
    run_replay --label "$label" --concurrency "$PD_CONCURRENCY" --queries "$dataset_dir/queries.jsonl" \
      --index-dir "$dataset_dir/index" --tokenizer "$MODEL" --tokenizer-revision "$MODEL_REVISION"
  else
    run_replay --label "$label" --concurrency "$PD_CONCURRENCY" \
      --frozen "$RUN_ROOT/1p3d-native/client/$label/frozen.jsonl"
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
    "${launch[@]}" -- bash "$RUN_ROOT/workload.sh" "$topology" "$mode"
    "$PYTHON" - "$RUN_ROOT/dataset-counts.json" "$run_dir/client" <<'PY'
import json
from pathlib import Path
import sys

counts = json.loads(Path(sys.argv[1]).read_text())
client = Path(sys.argv[2])
for name, row in counts.items():
    if name == "mtrag":
        continue  # The four domain receipts cover this total.
    receipt = json.loads((client / name.replace("/", "-") / "manifest.json").read_text())
    expected = row["requests"]
    if receipt["requested"] != expected or receipt["succeeded"] != expected:
        raise SystemExit(f"{name}: expected {expected} completed requests; inspect {client}")
PY
    if [[ "$topology/$mode" != 1p3d/native ]]; then
      for dataset in "${DATASETS[@]}"; do
        label="${dataset//\//-}"
        cmp -- "$RUN_ROOT/1p3d-native/client/$label/frozen.jsonl" "$run_dir/client/$label/frozen.jsonl"
      done
    fi
    "$PYTHON" scripts/vllm_pd_report.py --run-dir "$run_dir" --check --all-pairs
  done
done
```

The controlled fixture sends a cold request, an exact repeat to the same D, and a reordered passage request through every P→D pair. In the shared arm the controlled check verifies a full cold read, a warm D-local hit, fewer NIXL bytes on the repeat, and identical cold and warm output token IDs. The repeated evidence makes the prompt long enough to contain complete 256-token LMCache chunks. The trace and byte reconciliation checks precede full dataset replay. The final report uses the counters for the entire cohort. The complete dataset runs use the same replay manifests and a default concurrency of eight. Each cohort's frozen prompts are compared byte for byte with the first cohort's prompts.

### Historical native check (optional)

This optional check reuses the earlier five-conversation donor. It gives 5 warm-up and 36 measured requests at the original 9,216-token limit. If the prior checkout and its Git evidence are available, generate the frozen file below. An existing frozen file for the same donor also works. Run this from the same allocation after the FiQA smoke if you need the historical comparison. [Prior native input contract](../../context_pilot_paper/scripts/scheduler_v2_phase1_native_1p3d_inputs.py#L111)

```bash
PRIOR_REPO="${PRIOR_REPO:-$PD_HOME_REPO/../context_pilot_paper}"
PRIOR_SOURCE="e8cb79f3d668a795c44e2d878b2baf2c47444ad2"
if [[ -z "${DOCS8K_FROZEN:-}" ]]; then
  DOCS8K_FROZEN="$CHECK_ROOT/docs8k-frozen.json"
  "$PYTHON" "$PRIOR_REPO/scripts/scheduler_v2_phase1_native_1p3d_inputs.py" \
    --repo "$PRIOR_REPO" --source-commit "$PRIOR_SOURCE" \
    --conversation-count 5 --output "$DOCS8K_FROZEN"
fi
"$PYTHON" - "$DOCS8K_FROZEN" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    donor = json.load(stream)
assert donor["reference_commit"] == "bc733142a0789dc0b6b1dfb7876f50d4e8e05d0e"
assert (donor["warmup_request_count"], donor["measured_request_count"]) == (5, 36)
assert donor["engine_settings"]["max_model_len"] == 9216
print("Verified the prior five-conversation donor")
PY
run_dir="$CHECK_ROOT/1p3d-historical-native"
"$PYTHON" scripts/vllm_pd_cluster.py --run-dir "$run_dir" --topology 1p3d \
  --max-model-len 9216 --prefix-cache off --trace on \
  -- "$PYTHON" scripts/vllm_pd_replay.py replay --gateway http://127.0.0.1:8000 \
      --run-dir "$run_dir" --label prior-docs8k --model Qwen/Qwen3-4B-Instruct-2507 \
      --concurrency 5 --frozen "$DOCS8K_FROZEN"
"$PYTHON" scripts/vllm_pd_report.py --run-dir "$run_dir" --check --all-pairs
```

Inspect `summary.json.requests_by_phase` for 5 warm-ups and 36 measured requests. Use the historical run for exact-input and handoff verification on the new build.

## 3. Check the receipts in the allocation

| Artifact | What to verify |
| --- | --- |
| `run_manifest.json` and `endpoints.json` | vLLM fork point, current source commit and tree, working-tree change status, trace gate, correct model revision, four unique GPU UUIDs, and the exact topology, prefix settings, and ports. |
| `client/<label>/` | All query rows completed, reference prompts remained fixed on replay, and every evidence occurrence retained its document ID, chunk ID, version, and prompt token interval. |
| `smoke/*/smoke-check.json` | All 10 sampled FiQA requests completed with joined traces in both modes. Identifies the isolated requests and both complete conversations. |
| `dataset-counts.json` at the campaign root | Planned dev sets and MT-RAG domains match their expected request and conversation counts. |
| `trace/` and `timeline.json` | Request IDs join from replay or retrieval through routing, P scheduling and execution, D remote-KV wait, NIXL READ, D GPU decode, and client first and final tokens. |
| `requests.jsonl`, `summary.json`, and `metrics-{before,after}.json` | Local prefix matches by worker, selected external connector, GPU batch timings, completed NIXL bytes, prompt block coverage, and reconciliation. Native block overlaps belong to the shared-block accounting. |
| `native-metrics-{before,after}-{worker}.prom` | Raw native engine counters. The collector allows up to 30 seconds for the counters to catch up to traced NIXL completions. |
| `lmcache-status.json` | Verified mapping from LMCache's logical registration IDs to worker logs and physical GPU UUIDs. Expect four mappings in the shared arm. |
| `lmcache-metrics-{before,after}.prom` | Aggregate shared server cache and transfer phase counters. Check the request trace for the worker and selected connector before attributing a hit to a particular request. |
| `cleanup.json` and `logs/` | `remaining_process_groups` and `unverified_process_groups` are empty. Worker, gateway, workload, and metrics logs remain available for diagnosing an unsuccessful run. |

Inspect a cold and warm request from each topology. A warm P can restore a compatible prefix from LMCache or reuse its local prefix. A warm D can already hold part of the prompt and use NIXL for the blocks still missing. Check both the completed NIXL token range and actual bytes. A decoder can replay the last prompt token for first-token sampling, so the count of scheduled decode tokens is a separate measurement. [Instrumentation requirements](pd-instrumentation.md)

The default timeline shows the first 32 requests. To inspect one from a full dataset, run the same reporter with `--timeline-request` followed by its request ID from `requests.jsonl`.

The NIXL completion counter measures connector payload. The single-node route can use a local GPU transfer, so physical interface traffic requires its own device counter readout. A batch's GPU event duration belongs to that batch. Attribute token work to chunks using the mapped spans.

An interrupted cohort retains its partial evidence and closes the process groups it started. Inspect `logs/` and `cleanup.json`. For a rerun, choose a fresh cohort directory and use the existing frozen input receipts for prompts whose retrieval completed. Verify the allocation still owns the four GPUs with `nvidia-smi` before launching again.

Before leaving the allocation, copy the run and setup receipts to shared storage with enough quota. Set `EVIDENCESTORE_RESULTS_DIR` to that location.

```bash
: "${EVIDENCESTORE_RESULTS_DIR:?Set a writable results location on shared storage}"
archive_dir="$EVIDENCESTORE_RESULTS_DIR/$(basename "$RUN_ROOT")"
mkdir -p "$archive_dir"
rsync -a "$CHECK_ROOT/" "$archive_dir/setup/"
rsync -a "$RUN_ROOT/" "$archive_dir/run/"
```

## Measure the same build with tracing on and off

The same editable fork serves both runs. `--trace on` sets `EVIDENCESTORE_PD_TRACE=1` on each worker and enables request, scheduler, GPU batch, and NIXL completion events. `EVIDENCESTORE_TRACE_DIR` supplies their output directory. `--trace off` sets the source gate to `0`. The replay and router still capture client-side timings, and the reporter still reads the native aggregate NIXL counters. These runs report per-request engine work and payload as unavailable. `EVIDENCESTORE_TRACE_GPU=0` skips CUDA event timing within a trace-on run.

For an overhead check, replay the same frozen fixture in the same layout and prefix mode into two fresh run directories. Set `--trace on` for one and `--trace off` for the other. Match the source commit, model, GPU placement, and replay manifest, then compare client timing and native counter deltas. The earlier repo used the same runtime-gate pattern for its timing experiments. [Earlier measurement gate](../../context_pilot_paper/scripts/v4_es_matrix.sh#L46)

For a source comparison, commit each revision directly in `third_party/forks/vllm/` and run the same replay with the same trace settings for both builds. Use `python3 scripts/vllm_pd_prepare.py --export` to refresh the bundle and source receipt when transferring a revision to a different machine.
