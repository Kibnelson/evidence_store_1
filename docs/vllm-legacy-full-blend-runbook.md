# MuSiQue Full Blend smoke with the public in-process runtime

This runbook builds the vLLM `0.23.0` and LMCache `0.5.0` sources from the earlier Full Blend run in a fresh allocation. Each B worker owns an in-process LMCache CPU cache. D uses native NIXL to pull the full prompt KV with decode prefix reuse disabled. Start with **1B/3D**. Use **2B/2D** for two independent B caches, or run both layouts sequentially on the same four GPUs. The shared-server run uses its own launcher and `CBKVConnector`. [Source pins](../third_party/README.md#historical-full-blend-source) · [Shared-server runbook](vllm-native-pd-runbook.md#natural-musique-blend-smoke)

## Start from the login node

SSH to `research.cs.mcgill.ca` as `nbore`. If an earlier `srun` allocation is at its GPU prompt, exit that shell first. Paste this block at the login prompt. It verifies Slurm before starting a dedicated tmux server. The tmux pane requests four GPUs on `gpu-grad-01`, then performs the clone, build, data staging, replay, verification, cleanup, and archive inside the allocation.

```bash
layout=1b3d
trace=on

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
    bash "$source_repo/scripts/vllm_pd_legacy_blend_run.sh" "$1" "$2"; then
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
        --gpus=4 -c 16 --mem=192G -t 12:00:00 --pty /bin/bash -lc "$1" legacy-blend "$2" "$3"
      status=$?
      printf "Slurm step finished (exit %s).\n" "$status"
      exec /bin/bash --noprofile --norc -i
    ' legacy-blend "$gpu_command" "$layout" "$trace"
fi
```

Change `layout` to `2b2d` or `both` to change the GPU split. Keep `trace=on` for bring-up. A `trace=off` run retains the response receipts and aggregate NIXL counters. It skips request-level source tracing and the checks that depend on it.

Detach with `Ctrl+b` then `d`. To reattach, reconnect to the same login node and run:

```bash
tmux -L evidence-store-legacy-blend attach-session -t musique-legacy-blend
```

## What the command runs

1. The builder stages the committed experiment source under `/tmp/$USER`, restores the detached historical sources, and creates a fresh Python 3.12.14 venv. It installs the pinned vLLM CUDA 12.9 wheel libraries, builds editable LMCache with the site CUDA 12 compiler, and checks four visible GPUs and UCX GPU-memory support. The public build uses the `LMCacheConnectorV1` connector.
2. The workload stages the official MuSiQue development corpus, embeds it with `intfloat/e5-base-v2`, and builds the complete Faiss HNSW index. If `MUSIQUE_PREPARED` is exported before launch, the workload validates and uses that prepared corpus and index.
3. Each layout sends the same ten distinct MuSiQue questions once through live retrieval. The prompt wraps evidence in `¶`-delimited segments. Two synthetic requests per B then verify a cache fill followed by a reordered restore on the same B. The synthetic requests use their own documents and run after all natural questions.
4. The launcher captures native NIXL counters before and after replay and shuts down its worker groups. The report verifies completed responses and reconciles the NIXL read counts. Trace-on validation joins completed NIXL and LMCache movement and measured Blend recompute to versioned chunk spans. Natural requests can miss on a recurring chunk. The per-B controlled probe must produce a verified shifted restore.

The compute-node command can also run directly inside an existing idle four-GPU allocation from a clone of this repo:

```bash
bash scripts/vllm_pd_legacy_blend_run.sh 1b3d on
```

### Retry using a verified runtime

If a worker startup failed after runtime verification, use a fresh checkout of `main` and the original idle GPU allocation. Set `EVIDENCESTORE_LEGACY_RUNTIME` to the absolute runtime scratch directory printed during setup. The runner checks source compatibility, applies the worker import fix if needed, verifies the Python import, and reuses the built runtime and prepared MuSiQue index. Each retry has its own result directory and launcher log. A new allocation can reuse those files on the same node while the scratch directory remains available.

## Review the archive

A successful run saves the archive in `$HOME/evidence_store_legacy_reports` and prints its path and SHA-256. Set `EVIDENCESTORE_LEGACY_REPORT_DIR` to a writable persistent directory if the home quota is full. A build, replay, verification, or archive failure leaves the Slurm shell open with the scratch and log locations in the pane output.

Each archived `<topology>/blend/` directory contains:

- `legacy-blend-summary.json` with the natural and controlled counts, reuse checks, and NIXL reconciliation.
- `legacy-requests.jsonl` and `legacy-chunks.jsonl` from trace-on runs, with the gateway timeline, corpus and prompt stages, completed KV movement, and model work mapped to the supported token ranges.
- `client/`, `trace/`, counter snapshots, worker logs, and `cleanup.json` as supporting evidence.

When `layout=both`, `musique-pair.json` verifies matching natural prompts and evidence across layouts. Review the smoke before launching the full MuSiQue workload.

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
