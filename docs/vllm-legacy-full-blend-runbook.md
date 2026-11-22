# Full Blend replay with the public in-process runtime

This runbook builds the vLLM `0.23.0` and LMCache `0.5.0` sources from the earlier Full Blend run in a fresh allocation. Each B worker owns an in-process LMCache CPU cache. D uses native NIXL to pull the full prompt KV with decode prefix reuse disabled. Start with **1B/3D**. Use **2B/2D** for two independent B caches, or run both layouts sequentially on the same four GPUs. The shared-server run uses its own launcher and `CBKVConnector`. [Source pins](../third_party/README.md#historical-full-blend-source) · [Shared-server runbook](vllm-native-pd-runbook.md#natural-musique-blend-smoke)

## Start from the login node

SSH to `research.cs.mcgill.ca` as `nbore`. If an earlier `srun` allocation is at its GPU prompt, exit that shell first. Paste this block at the login prompt. It verifies Slurm before starting a dedicated tmux server. The tmux pane requests four GPUs on `gpu-grad-01`, then performs the clone, build, data staging, replay, verification, cleanup, and archive inside the allocation.

```bash
layout=1b3d
trace=on
workload=musique-smoke
prompt_policy=default

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf 'Return to the login shell before starting a new allocation.\n' >&2
elif module load slurm && srun --version; then
  gpu_command=$(cat <<'GPU'
umask 077
node_root="/tmp/$(id -un)"
if mkdir -p "$node_root" && [[ -O "$node_root" && ! -L "$node_root" ]] &&
    source_repo=$(mktemp -d "$node_root/legacy-blend-source-${SLURM_JOB_ID}.XXXXXX") &&
    git clone --depth=1 --branch main --no-tags \
      https://github.com/Kibnelson/evidence_store_1.git "$source_repo" &&
    bash "$source_repo/scripts/vllm_pd_legacy_blend_run.sh" "$1" "$2" "$3" "$4"; then
  printf 'Full Blend smoke verified and archived. Releasing the allocation.\n'
else
  failure=$?
  printf 'Full Blend stopped (exit %s). Inspect the output above. Type exit to release the job.\n' "$failure" >&2
  bash -l
  exit "$failure"
fi
GPU
  )
  tmux -L evidence-store-legacy-blend -f /dev/null new-session -s musique-legacy-blend \
    /bin/bash --noprofile --norc -c '
      srun -p all -A oana2026 -q research12h -N 1 -w gpu-grad-01 \
        --gpus=4 -c 16 --mem=192G -t 12:00:00 --pty /bin/bash -lc "$1" legacy-blend "$2" "$3" "$4" "$5"
      status=$?
      printf "Slurm step finished (exit %s).\n" "$status"
      exec /bin/bash --noprofile --norc -i
    ' legacy-blend "$gpu_command" "$layout" "$trace" "$workload" "$prompt_policy"
fi
```

Change `layout` to `2b2d` or `both` to change the GPU split. Keep `trace=on` for bring-up. A `trace=off` run retains the response receipts and aggregate NIXL counters. It skips request-level source tracing and the checks that depend on it.

Set `workload=musique-single-turn-40` or `workload=mtrag-multi-turn-repeat` to run the selected high-repeat workloads. The MT-RAG mode stages the pinned Human release and builds the full CLAPNQ index. `MTRAG_PREPARED` can point to an existing prepared CLAPNQ corpus and index. [Request selections and repeat counts](datasets/repeat-workloads.md)

Use `prompt_policy=seen-first` for MuSiQue or `prompt_policy=previous-turn-first` for the MT-RAG ordering test. MT-RAG also supports `prompt_policy=openclaw`. Keep `prompt_policy=default` for the baseline layout. [Prompt policy details and comparison commands](#compare-prompt-policies)

Detach with `Ctrl+b` then `d`. To reattach, reconnect to the same login node and run:

```bash
tmux -L evidence-store-legacy-blend attach-session -t musique-legacy-blend
```

## What the command runs

1. The builder stages the committed experiment source under `/tmp/$USER`, restores the detached historical sources, and creates a fresh Python 3.12.14 venv. It installs the pinned vLLM CUDA 12.9 wheel libraries, builds editable LMCache with the site CUDA 12 compiler, and checks four visible GPUs and UCX GPU-memory support. The public build uses the `LMCacheConnectorV1` connector.
2. The workload stages the official dataset, embeds the selected domain's complete corpus with `intfloat/e5-base-v2`, and builds the Faiss HNSW index. The runner can validate and reuse a prepared corpus and index from `MUSIQUE_PREPARED` or `MTRAG_PREPARED`.
3. Each layout sends the selected natural requests once through live retrieval. The default smoke has ten MuSiQue questions. The repeat-heavy workloads each have forty requests. The prompt wraps evidence in `¶`-delimited segments. Two synthetic requests per B then verify a cache fill followed by a reordered restore on the same B. The synthetic requests use their own documents and run after the natural replay.
4. The launcher captures native NIXL counters before and after replay and shuts down its worker groups. The report verifies completed responses and reconciles the NIXL read counts. Trace-on validation joins completed NIXL and LMCache movement and measured Blend recompute to versioned chunk spans. Natural requests can miss on a recurring chunk. The per-B controlled probe must produce a verified shifted restore.

The compute-node command can also run directly inside an existing idle four-GPU allocation from a clone of this repo:

```bash
bash scripts/vllm_pd_legacy_blend_run.sh 1b3d on
```

## Trace differential storage on MT-RAG

These runs use 1B/3D and `previous-turn-first`, so every turn goes through the same P cache. Choose `storage` for the 40-turn LMCache object trace or `pages` for turns 1–4 of CLAPNQ conversation `1534a095279f2cb888fb0bea17bd70da`. [Selected conversations](datasets/repeat-workloads.md)

### Start the allocation and run the trace

From the Slurm login node, start a dedicated tmux session. If you already have a tmux pane on the login node, run the allocation block at its prompt. If you already have an idle four-GPU allocation on `gpu-grad-01`, use its GPU shell.

```bash
tmux -L evidence-store-mtrag-storage -f /dev/null new-session -A -s mtrag-storage \
  /bin/bash -il
```

At the pane prompt, load and verify Slurm in the same shell that submits the allocation. If `module` is unavailable in an existing pane, run `exec bash -il` first and then paste the allocation block.

```bash
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf 'Already inside Slurm job %s. Use the compute-node step.\n' "$SLURM_JOB_ID"
elif module load slurm && srun --version; then
  srun -p all -A oana2026 -q research12h -N 1 -w gpu-grad-01 \
    --gpus=4 -c 16 --mem=192G -t 12:00:00 --pty bash -l
fi
```

Detach with `Ctrl+b` then `d`. Reattach from the same login node with `tmux -L evidence-store-mtrag-storage attach-session -t mtrag-storage`.

Run the following in the GPU shell. It builds a fresh runtime with the trace hooks and keeps the result archive and the launch log on node-local scratch. Set `MTRAG_PREPARED` to a matching prepared corpus and index to reuse one. The block defaults to the 40-turn object trace. For the four-turn native page smoke, set `kv_trace=pages` in the GPU shell before pasting it.

```bash
(
  set -euo pipefail
  umask 077
  : "${SLURM_JOB_ID:?Run inside the four-GPU allocation}"
  kv_trace="${kv_trace:-storage}"
  node_root="/tmp/$(id -un)"
  mkdir -p "$node_root"
  [[ -O "$node_root" && ! -L "$node_root" ]]
  df -h "$node_root"
  source_repo=$(mktemp -d "$node_root/mtrag-storage-source-$SLURM_JOB_ID.XXXXXX")
  git clone --depth=1 --branch main --no-tags \
    https://github.com/Kibnelson/evidence_store_1.git "$source_repo"
  unset EVIDENCESTORE_LEGACY_RUNTIME
  export EVIDENCESTORE_LEGACY_REPORT_DIR="$node_root/blend-storage-archives-$SLURM_JOB_ID"
  mkdir -p "$EVIDENCESTORE_LEGACY_REPORT_DIR"
  bash "$source_repo/scripts/vllm_pd_legacy_blend_run.sh" \
    1b3d on mtrag-multi-turn-repeat previous-turn-first "--trace-kv-$kv_trace" \
    2>&1 | tee "$EVIDENCESTORE_LEGACY_REPORT_DIR/launch.log"
)
```

The object trace can replay one exact prompt per P whose original request produced a patch candidate. Those verification requests have their own label. The page smoke measures native loads across the four natural turns and reports complete 16-token pages and boundary bytes separately. Both modes run two standard Full Blend probes to check the handoff and reconcile the counters. [Page and object trace measurements](vllm-legacy-full-blend-tracing.md#differential-storage-on-mt-rag)

### Recover a completed capture after a report error

If the replay finished and the report failed, stay on the compute node. Set `run_root` to the absolute path printed after `Run artifacts`. This block uses the saved traces to rebuild and archive the reports on CPU. It records the reporting revision alongside the original build and replay revisions.

```bash
(
  set -euo pipefail
  umask 077
  : "${run_root:?Set run_root to the printed Run artifacts path}"
  : "${SLURM_JOB_ID:?Use the GPU shell that ran the trace}"
  node_root="/tmp/$(id -un)"
  run_id="${run_root##*/}"
  runtime="${run_root%/repo/reports/runs/*}"
  cohort="$run_root/1p3d/blend"
  [[ "$run_id" == mtrag-multi-turn-repeat-legacy-full-blend-"$SLURM_JOB_ID"-* ]]
  [[ "$runtime" == "$node_root/"* && -O "$runtime" ]]
  test -s "$cohort/cleanup.json"
  report_dir="$node_root/blend-storage-archives-$SLURM_JOB_ID"
  mkdir -p "$report_dir"
  source_repo=$(mktemp -d "$node_root/mtrag-storage-recovery-$SLURM_JOB_ID.XXXXXX")
  git clone --depth=1 --branch main --no-tags \
    https://github.com/Kibnelson/evidence_store_1.git "$source_repo"
  python="$runtime/venv/bin/python"
  "$python" "$source_repo/scripts/vllm_pd_legacy_blend_report.py" --run-dir "$cohort" --trace on
  report_kind=$("$python" -c 'import json, sys; config = json.load(sys.stdin)["legacy_full_blend"]; print("page" if config.get("kv_page_trace_enabled") else "storage")' < "$cohort/run_manifest.json")
  if [[ ! -s "$cohort/legacy-kv-$report_kind-report.json" ]]; then
    "$python" "$source_repo/scripts/vllm_pd_legacy_${report_kind}_report.py" --run-dir "$cohort"
  fi
  archive="$report_dir/$run_id.tar.gz"
  if [[ ! -e "$archive" ]]; then
    git -C "$source_repo" rev-parse HEAD > "$runtime/check/$run_id-report-source.txt"
    temp_archive=$(mktemp "$report_dir/.recover-$run_id.XXXXXX")
    trap 'rm -f -- "$temp_archive"' EXIT
    tar -czf "$temp_archive" -C "$runtime" check -C "${run_root%/*}" "$run_id"
    tar -tzf "$temp_archive" >/dev/null
    mv -- "$temp_archive" "$archive"
  fi
  {
    sha256sum "$archive"
    printf 'MT-RAG KV %s trace archived at %s\n' "$report_kind" "$archive"
  } | tee -a "$report_dir/launch.log"
)
```

### Publish the verified packet

After the page or storage trace prints its archive path, use the same GPU shell to run the block below. It reads the archive path and digest from the launch log, rebuilds the review packet, and pushes that packet over SSH from a scratch clone. The compressed run archive stays on the node.

```bash
(
  set -euo pipefail
  umask 077
  : "${SLURM_JOB_ID:?Use the GPU shell that ran the trace}"
  node_root="/tmp/$(id -un)"
  archive_dir="$node_root/blend-storage-archives-$SLURM_JOB_ID"
  if ! read -r archive_sha256 archive < <(awk '
    length($1) == 64 && $1 ~ /^[0-9a-f]+$/ &&
      $2 ~ /\/mtrag-multi-turn-repeat-legacy-full-blend-[0-9]+-[0-9TZ]+\.tar\.gz$/ {
        print $1, $2
      }
  ' "$archive_dir/launch.log") || [[ "$archive" != "$archive_dir/"* || ! -s "$archive" ]]; then
    printf 'No completed MT-RAG storage archive in %s/launch.log\n' "$archive_dir" >&2
    exit 1
  fi
  publish_repo=$(mktemp -d "$node_root/mtrag-storage-publish-$SLURM_JOB_ID.XXXXXX")
  git clone --depth=1 --branch main --no-tags \
    https://github.com/Kibnelson/evidence_store_1.git "$publish_repo"
  cd "$publish_repo"
  result=$(python3 scripts/vllm_pd_legacy_blend_archive.py \
    --source "$archive" --sha256 "$archive_sha256")
  git remote set-url --push origin git@github.com:Kibnelson/evidence_store_1.git
  git add -- "$result"
  if ! git diff --cached --quiet -- "$result"; then
    git commit -m "Archive MT-RAG KV trace ${result##*/}" -- "$result"
  fi
  git push origin main
  printf 'Published %s at %s\n' "$result" "$(git rev-parse --short HEAD)"
  printf 'After pulling locally, run these from the repo root\n'
  printf 'python3 scripts/vllm_pd_legacy_blend_archive.py --verify %q\n' "$result"
  report="$result/1p3d/blend/legacy-kv-page-report.json"
  if [[ ! -f "$report" ]]; then
    report="$result/1p3d/blend/legacy-kv-storage-report.json"
  fi
  printf 'python3 -m json.tool %q\n' "$report"
)
```

### Pull and review locally

From your local `evidence_store_1` checkout, run `git pull --ff-only origin main`, then run the two commands printed by the publisher.

In `legacy-kv-storage-report.json`, start with `storage.gross_saved_bytes` for sampled savings including patch metadata. Then check `native_lookup_verification.nonidentical_delta_store_samples_with_verified_load` for changed targets reconstructed byte for byte on a later LMCache load. A report with zero savings is a valid result. A patch candidate without a verified load establishes a size opportunity only. [How the trace measures storage and restore](vllm-legacy-full-blend-tracing.md#differential-storage-on-mt-rag)

In `legacy-kv-page-report.json`, compare `storage.representations` for raw pages, references, and patches. Check `native_lookup_verification.all_layer_verified_events` and missing-base counts for the loads that were observed. The graph records the raw base and target page versions. Per-page rows retain native block mappings, token spans, and K/V changed-row counts by layer. [How the page trace identifies source and destination blocks](vllm-legacy-full-blend-tracing.md#native-page-smoke)

## Compare prompt policies

The fourth optional launcher argument controls client-side prompt assembly. `trace` separately controls instrumentation. The same Full Blend workers perform prefill and decode, with a full native NIXL handoff and counters reconciled per decoder. Controlled probes use the original prompt layout in every mode.

| `prompt_policy` | Workload | Prompt assembly |
| --- | --- | --- |
| `default` | Any | Instructions, then `¶`-delimited evidence in Faiss order, then saved conversation history and the current question. |
| `seen-first` | MuSiQue | Uses the default layout and selects evidence by the original retrieval ranking and prompt budget. Among the selected chunks, orders chunks seen on earlier successful requests by their first appearance during serial replay, then places new chunks in Faiss order. Receipts record original retrieval ranks. |
| `previous-turn-first` | `mtrag-multi-turn-repeat` | Uses the legacy layout with evidence before history and question. Orders selected chunks retained by the previous successful turn of the same conversation in that turn’s prompt order. Other chunks retained by earlier successful requests follow in order of first appearance in the replay. Unseen chunks follow in Faiss order. Selection uses the original Faiss ranks and prompt budget. Receipts record the original retrieval ranks. |
| `openclaw` | `mtrag-multi-turn-repeat` | Instructions `¶` saved reference history and current question `¶` retrieved documents in Faiss order `¶` answer cue. |

For `seen-first`, earlier inclusion is an ordering signal across the replay. In 2B/2D, an earlier request can have used the other producer's cache. Evaluate actual reuse from completed host-to-device restores and traced Blend recompute.

For `openclaw`, the published demo uses two turns from MT-RAG Cloud and places the current query between the system prompt and its retrieved supporting document. Our selected CLAPNQ workload retains the saved reference history before the question. The pinned legacy connector ends layerwise matching at the first missing segment, so a changed history/question segment can prevent restores of documents that follow it. [LMCache OpenClaw demo](https://blog.lmcache.ai/en/2026/04/01/accelerating-openclaw-agents-with-cacheblend/) · [Legacy connector and source pins](../third_party/README.md#historical-full-blend-source)

### Run both pairs in one allocation

On the login node, open a dedicated tmux session for the GPU allocation.

```bash
module load slurm && srun --version &&
  tmux -L evidence-store-prompt-gates -f /dev/null new-session -s blend-policies \
    /bin/bash --noprofile --norc -c '
      srun -p all -A oana2026 -q research12h -N 1 -w gpu-grad-01 \
        --gpus=4 -c 16 --mem=192G -t 12:00:00 --pty bash -l
      exec /bin/bash --noprofile --norc -i
    '
```

Inside the GPU shell, paste this block. An exported `EVIDENCESTORE_LEGACY_RUNTIME` points to a verified scratch runtime on this node. With the variable unset, the block builds once. Each arm starts fresh workers with the same runtime and prepared index for that workload. Raw archives go to node-local scratch by default.

```bash
(
  set -euo pipefail
  umask 077
  : "${SLURM_JOB_ID:?Run inside the four-GPU allocation}"
  node_root="/tmp/$(id -un)"
  mkdir -p "$node_root"
  [[ -O "$node_root" && ! -L "$node_root" ]]
  df -h "$node_root"
  source_repo=$(mktemp -d "$node_root/blend-policy-source-$SLURM_JOB_ID.XXXXXX")
  git clone --depth=1 --branch main --no-tags \
    https://github.com/Kibnelson/evidence_store_1.git "$source_repo"
  if [[ -z "${EVIDENCESTORE_LEGACY_RUNTIME:-}" ]]; then
    activation_receipt=$(mktemp "$node_root/blend-policy-activation-$SLURM_JOB_ID.XXXXXX")
    trap 'rm -f -- "$activation_receipt"' EXIT
    bash "$source_repo/scripts/vllm_pd_legacy_blend_runtime.sh" "$activation_receipt"
    source "$(<"$activation_receipt")"
    export EVIDENCESTORE_LEGACY_RUNTIME="$PD_EXEC_ROOT"
  fi
  export EVIDENCESTORE_LEGACY_REPORT_DIR="${EVIDENCESTORE_LEGACY_REPORT_DIR:-$node_root/blend-policy-archives-$SLURM_JOB_ID}"
  runner="$source_repo/scripts/vllm_pd_legacy_blend_run.sh"
  bash "$runner" 1b3d on musique-single-turn-40 default
  bash "$runner" 1b3d on musique-single-turn-40 seen-first
  bash "$runner" 1b3d on mtrag-multi-turn-repeat default
  bash "$runner" 1b3d on mtrag-multi-turn-repeat openclaw
)
```

Reattach on the same login node with `tmux -L evidence-store-prompt-gates attach-session -t blend-policies`. Compare one workload at a time from a current source checkout. Set `default_cohort` and `gated_cohort` to the two `1p3d/blend` directories under the printed run artifact paths, or use the corresponding published review directories.

```bash
python3 scripts/vllm_pd_legacy_blend_report.py --compare "$default_cohort" "$gated_cohort"
```

The comparison matches requests and versioned chunks and reports ordinary P tokens, Blend attention/MLP token-layer positions, completed LMCache restore and store bytes, and NIXL payload. Evidence-exclusive NIXL bytes omit blocks shared with another span. The whole-request total includes those blocks. A successful `openclaw` run can have zero document restores while its controlled probes verify that the trace records reuse. [Publish the review packets](#publish-the-review-packet-to-git) while the scratch archives are available.

Check the p50 and p95 prompt-render time, client time to first token (TTFT), dispatch-to-first-token time, and prefill HTTP time. The report also computes gated-minus-default differences for matching request keys. Client TTFT includes retrieval and prompt assembly. Dispatch-to-first-token time starts after those stages. These timings include the synchronization used by trace-on runs.

### Test MT-RAG only

From a current checkout inside an idle four-GPU allocation, run the block below. Both arms replay the same 40 CLAPNQ turns in order with tracing on. The first arm prepares the CLAPNQ index if needed, and the second reuses it. Set `MTRAG_PREPARED` to use an existing prepared CLAPNQ corpus and index. The block can reuse an exported `EVIDENCESTORE_LEGACY_RUNTIME` that points to a verified runtime on this node. Archives default to node-local scratch.

```bash
(
  set -euo pipefail
  umask 077
  : "${SLURM_JOB_ID:?Run inside the four-GPU allocation}"
  source_repo=$(git rev-parse --show-toplevel)
  node_root="/tmp/$(id -un)"
  mkdir -p "$node_root"
  [[ -O "$node_root" && ! -L "$node_root" ]]
  if [[ -z "${EVIDENCESTORE_LEGACY_RUNTIME:-}" ]]; then
    activation_receipt=$(mktemp "$node_root/blend-mtrag-activation-$SLURM_JOB_ID.XXXXXX")
    trap 'rm -f -- "$activation_receipt"' EXIT
    bash "$source_repo/scripts/vllm_pd_legacy_blend_runtime.sh" "$activation_receipt"
    source "$(<"$activation_receipt")"
    export EVIDENCESTORE_LEGACY_RUNTIME="$PD_EXEC_ROOT"
  fi
  export EVIDENCESTORE_LEGACY_REPORT_DIR="${EVIDENCESTORE_LEGACY_REPORT_DIR:-$node_root/blend-mtrag-archives-$SLURM_JOB_ID}"
  runner="$source_repo/scripts/vllm_pd_legacy_blend_run.sh"
  bash "$runner" 1b3d on mtrag-multi-turn-repeat default
  bash "$runner" 1b3d on mtrag-multi-turn-repeat previous-turn-first
)
```

Use the `--compare` command above with the two printed MT-RAG cohort paths. Check matching evidence selections and verified LMCache restores on later turns. The optional `openclaw` comparison remains available from the same workload and index.

### Repeat MT-RAG and MuSiQue with delta estimates

Use the same four-GPU tmux allocation as above. Paste this block inside the GPU shell. It clones `main` to node-local scratch, so you can run it from your home directory. It reuses a verified runtime if `EVIDENCESTORE_LEGACY_RUNTIME` is set and builds one otherwise. Both workloads replay their same 40 selected requests with fresh 1B/3D workers. Set `MTRAG_PREPARED` and `MUSIQUE_PREPARED` if their prepared indexes are available.

```bash
(
  set -euo pipefail
  umask 077
  : "${SLURM_JOB_ID:?Run inside the four-GPU allocation}"
  node_root="/tmp/$(id -un)"
  mkdir -p "$node_root"
  [[ -O "$node_root" && ! -L "$node_root" ]]
  df -h "$node_root"
  source_repo=$(mktemp -d "$node_root/blend-estimate-source-$SLURM_JOB_ID.XXXXXX")
  git clone --depth=1 --branch main --no-tags \
    https://github.com/Kibnelson/evidence_store_1.git "$source_repo"
  test -r "$source_repo/scripts/vllm_pd_legacy_delta_estimate.py"
  if [[ -z "${EVIDENCESTORE_LEGACY_RUNTIME:-}" ]]; then
    activation_receipt=$(mktemp "$node_root/blend-estimate-activation-$SLURM_JOB_ID.XXXXXX")
    trap 'rm -f -- "$activation_receipt"' EXIT
    bash "$source_repo/scripts/vllm_pd_legacy_blend_runtime.sh" "$activation_receipt"
    source "$(<"$activation_receipt")"
    export EVIDENCESTORE_LEGACY_RUNTIME="$PD_EXEC_ROOT"
  fi
  export EVIDENCESTORE_LEGACY_REPORT_DIR="${EVIDENCESTORE_LEGACY_REPORT_DIR:-$node_root/blend-estimate-archives-$SLURM_JOB_ID}"
  runner="$source_repo/scripts/vllm_pd_legacy_blend_run.sh"
  bash "$runner" 1b3d on mtrag-multi-turn-repeat previous-turn-first --estimate-delta
  bash "$runner" 1b3d on musique-single-turn-40 seen-first --estimate-delta
)
```

The workers send the full NIXL payload. With `--estimate-delta`, the runner saves `legacy-delta-estimate.json` and `legacy-delta-estimate.jsonl` under each printed `1p3d/blend/` cohort. The first file has totals, and the second links each natural request to its measured bytes, READ interval, and chunk candidates. The scenarios assume 10%, 50%, 90%, or 100% of covered KV changes. Both workloads report the scope where D has a base for all verified repeat restores. MT-RAG also reports the narrower scope limited to chunks in the previous turn of the same conversation. These assumptions give a projected byte saving and ideal bandwidth-only speedup. The live timings still come from full transfers. [Formula sheet](kv-delta-speedup.md)

To generate the same estimates from a published cohort without a GPU, run `python3 scripts/vllm_pd_legacy_delta_estimate.py "$cohort" --output "$(mktemp -d)"`. The publisher verifies and includes the estimate summary and compressed request log when the launcher records the opt-in. [Publish the review packet](#publish-the-review-packet-to-git)

## Retry using a verified runtime

If a worker startup failed after runtime verification, use a fresh checkout of `main` and the original idle GPU allocation. Set `EVIDENCESTORE_LEGACY_RUNTIME` to the absolute runtime scratch directory printed during setup. The runner checks source compatibility, applies the worker import fix if needed, verifies the Python import, and reuses the built runtime and any prepared index for the selected workload. Each retry has its own result directory and launcher log. A new allocation can reuse those files on the same node while the scratch directory remains available.

## Review the archive

[Full Blend tracing for compute and KV movement](vllm-legacy-full-blend-tracing.md) explains the source hooks, per-chunk attribution, and a verified MuSiQue example.

A successful run saves the archive in `$HOME/evidence_store_legacy_reports` and prints its path and SHA-256. Set `EVIDENCESTORE_LEGACY_REPORT_DIR` to a writable persistent directory if the home quota is full. A build, replay, verification, or archive failure leaves the Slurm shell open with the scratch and log locations in the pane output.

Each archived `<topology>/blend/` directory contains:

- `legacy-blend-summary.json` with the natural and controlled counts, reuse checks, and NIXL reconciliation.
- `legacy-requests.jsonl` and `legacy-chunks.jsonl` from trace-on runs, with the gateway timeline, corpus and prompt stages, completed KV movement, and model work mapped to the supported token ranges.
- `client/`, `trace/`, counter snapshots, worker logs, and `cleanup.json` as supporting evidence.

When `layout=both`, `musique-pair.json` or `mtrag-pair.json` verifies matching natural prompts and evidence across layouts. Review the smoke before launching the full dataset workload.

## Publish the review packet to Git

From a scratch checkout on `main`, set `archive` to the `.tar.gz` path and `archive_sha256` to the digest printed at the end of the smoke. The publisher verifies the tarball, recomputes the measurement report from the traces, and stages the summary, request and chunk measurements, counters, response hashes, and source receipt in `reports/smoke/<run-id>/`. Prompt and completion content and build logs stay in the original tarball.

```bash
if result=$(python3 scripts/vllm_pd_legacy_blend_archive.py \
      --source "$archive" --sha256 "$archive_sha256"); then
  git remote set-url --push origin git@github.com:Kibnelson/evidence_store_1.git &&
    git add -- "$result" &&
    git commit -m "Archive Full Blend smoke ${result##*/}" -- "$result" &&
    git push origin main
fi
```

After pulling that result commit, verify the review packet with `python3 scripts/vllm_pd_legacy_blend_archive.py --verify "$result"` from the repository root.
