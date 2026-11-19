# Native vLLM PD runbook

This runbook compares a full native NIXL handoff with **one shared LMCache MP server for P and D** and native decode prefix reuse enabled. Both layouts use vLLM's native NIXL pull. A cold D pulls the full prompt KV. A warm D can use blocks already resident in its GPU cache and pull the missing part. The LMCache runs use the public prefix cache connector. [LMCache multiprocess deployment](https://docs.lmcache.ai/mp/deployment.html)

The shared arm configures NIXL first and LMCache second on both roles. Native `MultiConnector` selects the first connector that reports a positive external match. The trace records the selected connector, D's local prefix, and completed NIXL bytes. This shows whether a request actually used LMCache for D. The manual PD recipe uses dedicated servers and the operator documents one per node. Verify the shared single-node configuration with the controlled smoke before running the datasets. [Pinned selector](https://github.com/vllm-project/vllm/blob/b24eda7127a9c5c317f699db3f580135eec7322b/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py) · [LMCache PD operator](https://docs.lmcache.ai/mp/operator.html#pd-disaggregation) · [Manual PD recipe](https://github.com/LMCache/LMCache/blob/042d47abdef0d460cf2cb0d5720e8a59280cda8a/docs/source/mp/disaggregated_prefill.rst)

The default is **1P/3D**. The **2P/2D** arm reassigns one GPU from decode to prefill. Each worker uses one GPU with TP=1. The gateway sends a one-token request to P, receives the native KV coordinates, selects D, and relays D's completion stream. Force all P→D pairs in the controlled tests. Use the gateway's round-robin routing for the datasets.

Run section 0 on the cluster login node. Run section 1 onward from the compute-node shell of **one single-node Slurm allocation with four GPUs**. The launcher supervises its workers, gateway, and workload inside that allocation. Startup, both layouts, dataset ingestion and replay, shutdown, and reporting all use the same allocation. The earlier run card supplies the model, runtime settings, and frozen input format. [Prior native PD run card](../../context_pilot_paper/docs/evidencestore/implementation/scheduler_v2_phase2_native_and_three_modes_run_card.md)

For the new per-chunk instrumentation, follow the [MuSiQue chunk smoke from a fresh login shell](#musique-chunk-smoke-from-a-fresh-login-shell). For the original FiQA smoke, follow [login and allocation](#0-prepare-the-login-shell-and-request-the-allocation), [runtime setup](#1-prepare-the-runtime-and-the-inputs), and [FiQA smoke](#one-dataset-smoke). After both FiQA smoke receipts pass, you can [save the receipts](#3-check-the-receipts-in-the-allocation) or continue to the [full dataset inputs](#dataset-inputs).

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

## MuSiQue chunk smoke from a fresh login shell

Start here when the previous job has ended and `pwd` prints `/home/2023/nbore/evidence_store_1` at the login prompt. This smoke runs seven live MuSiQue requests and four controlled probes in each 2P/2D cohort. It builds the trace-enabled LMCache runtime inside the new allocation and checks both cohorts before the full MuSiQue rerun. [What the smoke must prove](datasets/musique.md#what-must-pass)

### Request a new allocation

From `bash-5.1$` on the login node, load Slurm and request four GPUs on the same node as the earlier run.

```bash
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf 'Already inside job %s. Continue with the compute-node step.\n' "$SLURM_JOB_ID"
elif module load slurm && srun --version; then
  srun -p all -A oana2026 -q research12h -N 1 -w gpu-grad-01 \
    --gpus=4 -c 16 --mem=192G -t 12:00:00 --pty bash -l ||
    printf 'Slurm step ended with an error. Check the output above.\n' >&2
fi
```

### Clone and build on the compute node

Wait for the `gpu-grad-01` shell. The checkout and build below use node-local scratch, including Git objects, the venv, and dependency caches. This keeps the setup outside the home checkout that hit its quota during publication.

```bash
if [[ -n "${SLURM_JOB_ID:-}" ]] &&
    node_scratch="/tmp/$(id -un)" &&
    mkdir -p "$node_scratch" &&
    [[ -O "$node_scratch" && ! -L "$node_scratch" ]] &&
    smoke_source=$(mktemp -d "$node_scratch/musique-source-${SLURM_JOB_ID}.XXXXXX") &&
    git clone --depth=1 --branch main --no-tags \
      https://github.com/Kibnelson/evidence_store_1.git "$smoke_source" &&
    test -s "$smoke_source/third_party/lmcache-pd.bundle" &&
    test -r "$smoke_source/scripts/vllm_pd_chunk_smoke.sh" &&
    git -C "$smoke_source" rev-parse --short HEAD &&
    env -u CUDA_HOME EVIDENCESTORE_LMCACHE_TRACE=1 \
      bash "$smoke_source/scripts/vllm_pd_runtime.sh"; then
  printf 'Run the exact source command printed above in this allocation shell.\n'
else
  printf 'Setup stopped. Check the error above and the runtime check/setup.log if it was created.\n' >&2
fi
```

Setup succeeds when it prints `Runtime verified` followed by a `source` command. Run that exact command in the `gpu-grad-01` shell. It activates the new scratch runtime and sets `REPO`, `PYTHON`, and `PD_EXEC_ROOT`.

### Run the MuSiQue smoke

This uses the previous MuSiQue index if its files are still on `gpu-grad-01`. Otherwise it stages the pinned development split and builds the full MuSiQue index before replaying the small sample. `MUSIQUE_PREPARED` can point to another retained full index and its prepared inputs.

```bash
prior_musique="/tmp/$(id -un)/espd-30450.wi1VL0/repo/reports/runs/native-vllm-30453-20261119T031129Z/data/musique"
retained_musique="${MUSIQUE_PREPARED:-$prior_musique}"
if (
  unset MUSIQUE_PREPARED MUSIQUE_DEV
  if [[ -r "$retained_musique/manifest.json" &&
        -r "$retained_musique/corpus.jsonl" &&
        -r "$retained_musique/queries.jsonl" &&
        -r "$retained_musique/index/manifest.json" &&
        -r "$retained_musique/index/index.faiss" &&
        -r "$retained_musique/index/chunks.jsonl" ]]; then
    MUSIQUE_PREPARED="$retained_musique" bash "$REPO/scripts/vllm_pd_chunk_smoke.sh"
  elif musique_dev=$("$PYTHON" "$REPO/scripts/vllm_pd_stage_inputs.py" musique "$PD_EXEC_ROOT/chunk-inputs"); then
    MUSIQUE_DEV="$musique_dev" bash "$REPO/scripts/vllm_pd_chunk_smoke.sh"
  else
    exit 1
  fi
); then
  printf 'Both MuSiQue smoke cohorts passed. Review the receipts before the full rerun.\n'
else
  printf 'Smoke stopped. The allocation shell is still open for inspection.\n' >&2
fi
```

Use the path printed after `MuSiQue chunk smoke artifacts`. Under that run, review `2p2d-native/chunk-review/summary.json` and `2p2d-shared/chunk-review/summary.json`. The `probe_cases` section shows prefill token work, completed NIXL bytes, and LMCache staging bytes for the four controlled requests. The paired-input check verifies the seven live requests before the driver prints `MuSiQue chunk smoke passed`. Set `RUN_ROOT` to the printed artifact path before copying the run and setup receipts with [section 3](#3-check-the-receipts-in-the-allocation). [MuSiQue acceptance checks](datasets/musique.md#what-must-pass)

## Full MuSiQue single-pass replay

Use the four-GPU allocation and verified trace-enabled runtime from the MuSiQue smoke. `--full` runs `2p2d-shared`. Add `--paired` to run the native cohort first and reproduce the paired comparison. Each cohort sends all 2,417 development questions once, in dataset order, with concurrency 1 and live Faiss retrieval. The gateway uses its normal routing. [MuSiQue analysis](datasets/musique.md#full-single-pass-review)

If the verified runtime is still active, refresh the analysis and replay scripts from a node-local clone. The receipt comparisons verify that the existing runtime uses the same dependency pins and vLLM and LMCache forks. If the allocation ended, get a new one and build from current `main` using the allocation and build steps above, then continue to the replay command.

```bash
if [[ -n "${SLURM_JOB_ID:-}" && -n "${PD_EXEC_ROOT:-}" && -n "${REPO:-}" ]] &&
    run_source=$(mktemp -d "$PD_EXEC_ROOT/musique-source.XXXXXX") &&
    git clone --depth=1 --branch main --no-tags \
      https://github.com/Kibnelson/evidence_store_1.git "$run_source" &&
    cmp "$REPO/third_party/sources.json" "$run_source/third_party/sources.json" &&
    cmp "$REPO/third_party/vllm-fork.json" "$run_source/third_party/vllm-fork.json" &&
    cmp "$REPO/third_party/lmcache-pd-fork.json" "$run_source/third_party/lmcache-pd-fork.json" &&
    git -C "$run_source" archive HEAD scripts/vllm_pd_chunk_smoke.sh \
      scripts/vllm_pd_chunk_smoke_inputs.py scripts/vllm_pd_musique_repeat_report.py \
      scripts/vllm_pd_archive_musique.py src/evidencestore/pd_chunk_report.py |
      tar -xf - -C "$REPO"; then
  git -C "$run_source" rev-parse HEAD | tee "$CHECK_ROOT/musique-runner-source.txt"
else
  printf 'Refresh stopped. Check the error before launching the full replay.\n' >&2
fi
```

Launch from the allocation shell. The runner reuses the earlier full MuSiQue index when present. It stages the pinned input and builds the index in scratch otherwise. Set `MUSIQUE_PREPARED` and optionally `MUSIQUE_INDEX` to reuse another copy.

```bash
prior_musique="/tmp/$(id -un)/espd-30450.wi1VL0/repo/reports/runs/native-vllm-30453-20261119T031129Z/data/musique"
retained_musique="${MUSIQUE_PREPARED:-$prior_musique}"
if [[ -r "$retained_musique/manifest.json" && -r "$retained_musique/corpus.jsonl" &&
      -r "$retained_musique/queries.jsonl" ]]; then
  MUSIQUE_PREPARED="$retained_musique" bash "$REPO/scripts/vllm_pd_chunk_smoke.sh" --full ||
    printf 'Replay stopped. Inspect the printed run directory.\n' >&2
elif musique_dev=$("$PYTHON" "$REPO/scripts/vllm_pd_stage_inputs.py" musique "$PD_EXEC_ROOT/chunk-inputs"); then
  MUSIQUE_PREPARED= MUSIQUE_DEV="$musique_dev" bash "$REPO/scripts/vllm_pd_chunk_smoke.sh" --full ||
    printf 'Replay stopped. Inspect the printed run directory.\n' >&2
fi
```

Set `RUN_ROOT` to the path printed after `MuSiQue chunk full artifacts`. Count response receipts in `client/musique/responses.jsonl` or follow `logs/workload.log` inside the active cohort. Review the shared run's `chunk-review/summary.json` and `chunks.jsonl` for repeated chunks and their previous positions. With `--paired`, the additional `repeat-summary.json` verifies matched prompts, and `repeated-chunks.jsonl` compares work and exclusive KV bytes across the cohorts. Use `chunk-review/kv-transfers.jsonl` for blocks crossing evidence boundaries.

Keep the complete raw run on shared storage with the [copy commands in section 3](#3-check-the-receipts-in-the-allocation). To publish a paired run, prepare the compressed measurements in the scratch clone from the refresh step. If you built a fresh runtime from `main`, use its node-local `PD_HOME_REPO` as the review checkout. Set the checkout's Git identity as for the earlier dataset pushes.

```bash
review_dir="reports/datasets/$(basename "$RUN_ROOT")/details/musique"
publish_repo="${run_source:-$PD_HOME_REPO}"
if git -C "$publish_repo" var GIT_AUTHOR_IDENT &&
    git -C "$publish_repo" remote set-url --push origin git@github.com:Kibnelson/evidence_store_1.git &&
    git -C "$publish_repo" pull --ff-only origin main &&
    "$PYTHON" "$REPO/scripts/vllm_pd_archive_musique.py" \
      --run-root "$RUN_ROOT" --output "$publish_repo/$review_dir" &&
    git -C "$publish_repo" add -- "$review_dir" &&
    git -C "$publish_repo" commit -m "Archive MuSiQue single-pass chunk measurements" -- "$review_dir" &&
    git -C "$publish_repo" push origin main; then
  printf 'MuSiQue measurements pushed.\n'
else
  printf 'Publication stopped. The allocation and its measurements are still available.\n' >&2
fi
```

The archive manifest provides file sizes and checksums for verification after pulling. If the final push failed after the commit, retry `git -C "$publish_repo" push origin main` from the allocation.

### Archive native after stopping the shared replay

In a second shell on the allocated GPU node, set `RUN_ROOT` to the full replay's printed path and signal the exact shared launcher. The guarded launch above returns to its allocation shell after the supervisor writes `cleanup.json`. Check that `remaining_process_groups` and `unverified_process_groups` are empty before reusing the GPUs.

```bash
mapfile -t shared_pids < <(ps -ww -u "$(id -un)" -o pid=,args= |
  awk -v dir="$RUN_ROOT/2p2d-shared" \
    '$3 == "scripts/vllm_pd_cluster.py" && $4 == "--run-dir" && $5 == dir {print $1}')
if (( ${#shared_pids[@]} == 1 )); then
  kill -TERM "${shared_pids[0]}"
else
  printf 'Found %s matching shared launchers\n' "${#shared_pids[@]}"
fi
```

The native `datasets/musique/summary.json` covers dataset aggregates. Its `chunk-review` contains the individual request, chunk, and KV-transfer measurements. Archive the completed native cohort from a scratch checkout using the verified Python environment.

```bash
review_dir="reports/datasets/$(basename "$RUN_ROOT")/details/musique-native"
if review_source=$(mktemp -d "$PD_EXEC_ROOT/musique-native-review.XXXXXX") &&
    git clone --depth=1 --branch main --no-tags \
      https://github.com/Kibnelson/evidence_store_1.git "$review_source" &&
    "$PYTHON" "$review_source/scripts/vllm_pd_archive_musique.py" \
      --run-root "$RUN_ROOT" --cohort 2p2d-native --output "$review_source/$review_dir"; then
  printf 'Native review saved in %s\n' "$review_source/$review_dir"
fi
```

For Git publication, set the scratch checkout's Git identity as in the paired workflow, set its SSH push URL, and commit only `"$review_dir"`. The archive manifest verifies the native cohort, including the 2,417 requests and reconciled NIXL payload. Use the original GPU allocation shell for the [natural MuSiQue Blend smoke](#natural-musique-blend-smoke) after cleanup and the Blend plugin preflight.

## Switch prefix prefill to Blend on one LMCache server

The smoke covers **1P/3D → 1B/3D** and **2P/2D → 2B/2D**. Within each layout one LMCache MP `blend` server stays up while the prefix workers store KV and the Blend workers restore it. Each phase starts vLLM workers on the same GPU slots. The prefix mode uses `LMCacheMPConnector`, and the Blend mode uses the separate `CBKVConnector` plugin. D pulls the full prompt KV over native NIXL in both modes with D prefix reuse off. [Measurement plan](slide-4-pd-measurement-plan.md#seed-on-prefix-prefill-and-reuse-on-blend)

The controlled inputs use the previous Full Blend prompt layout with `¶` separating the evidence segments. Prepare and validate the two layouts without a GPU allocation using a Python environment with the replay tokenizer dependency installed.

```bash
python3 scripts/vllm_pd_blend_smoke.py --prepare-only \
  --run-root "/tmp/musique-blend-prompts-$(date -u +%Y%m%dT%H%M%SZ)"
```

For the GPU smoke, use an idle four-GPU allocation and a verified runtime from current `main` in node-local scratch. Build the runtime with `EVIDENCESTORE_LMCACHE_TRACE=1` and use that build for both trace settings. Install the approved `lmcache-cacheblend` plugin compatible with the pinned vLLM and LMCache. Record the plugin version and artifact digest with the runtime receipt. Use the previous in-process run card for a historical 1B/3D reproduction. The previous run's separate prefix and Blend caches cannot establish this cross-mode restore. [LMCache plugin setup](https://docs.lmcache.ai/mp/operator.html#deploying-a-cacheblendengine) · [Previous Full Blend run card](../../context_pilot_paper/docs/evidencestore/implementation/scheduler_v2_phase2_native_and_three_modes_run_card.md#full-blend-flow-and-pressure-behavior)

```bash
if "$PYTHON" "$REPO/scripts/vllm_pd_blend_smoke.py" --trace on \
    --run-root "$REPO/reports/runs/blend-smoke-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"; then
  printf 'Shared-cache Blend smoke passed.\n'
else
  printf 'Blend smoke stopped. Inspect the error and the printed run directory.\n' >&2
fi
```

Add `--topology 1p3d` or `--topology 2p2d` to run an individual layout. `--trace on` is the default. It records the client and gateway timeline, vLLM scheduler work and CUDA batch times, native NIXL transfers, and LMCache lookups and completions. The exact probe verifies prefix restore. The shifted probe uses evidence from a separate P seed and requires a non-prefix hit and successful scatter. In each topology directory, `lmcache-request-trace.jsonl` joins cache events to native request IDs and maps completed host-to-GPU payload and old and new token ranges to evidence spans. Per-layer repair positions inside the private connector remain unavailable.

Run the same command in a fresh output directory with `--trace off` to disable these detailed traces. Check the output hashes and cases in `blend-summary.json` and the aggregate NIXL counters and LMCache reuse checks in either mode. Trace-off reports leave request-level transfer and scheduler measurements unavailable.

## Natural MuSiQue Blend smoke

Use the same trace-enabled runtime and `CBKVConnector` plugin as the controlled smoke. This run starts a fresh LMCache server for **1B/3D** and another fresh server for **2B/2D**. The Blend workers in each layout share their server. Both layouts replay ten distinct MuSiQue questions once through live Faiss retrieval and normal gateway routing. The questions form two groups that retrieved recurring chunks in the [earlier MuSiQue archive](https://github.com/Kibnelson/evidence_store_1/tree/e1c8b5096cf4b9fc9721614b95dedb7280300bf5/reports/datasets/native-vllm-30453-20261119T031129Z/details/musique). Evidence uses the `¶` prompt format. The first ordinary requests populate the cache.

Run from the active venv in an idle four-GPU allocation. This uses the full MuSiQue corpus index already prepared for the prefix campaign. Set `MUSIQUE_PREPARED` if you have another copy. Use `--index-dir` if its Faiss index is stored separately.

```bash
prepared_musique="${MUSIQUE_PREPARED:-/tmp/$(id -un)/espd-30450.wi1VL0/repo/reports/runs/native-vllm-30453-20261119T031129Z/data/musique}"
if "$PYTHON" "$REPO/scripts/vllm_pd_blend_smoke.py" --trace on \
    --musique-prepared "$prepared_musique" \
    --run-root "$REPO/reports/runs/musique-natural-blend-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"; then
  printf 'Natural Blend smoke passed.\n'
else
  printf 'Smoke stopped. Inspect the error and printed run directory.\n' >&2
fi
```

Add `--prepare-only` to select and record the input questions without launching GPU workers. Add `--topology 1p3d` or `--topology 2p2d` to test one layout at a time. Review `natural-blend-summary.json` and `lmcache-request-trace.jsonl` under each layout. The trace-on smoke requires a completed shifted restore tied to evidence from an earlier question and a reconciled native NIXL handoff. `musique-pair.json` verifies matched prompts and evidence across the two layouts. Confirm these receipts before running all 2,417 questions. For a counter-only check, use `--trace off` in a fresh output directory.

## 1. Prepare the runtime and the inputs

If setup failed and the `gpu-grad-01` prompt is still open with a Slurm job ID, update the shared checkout from that shell and retry section 1.2 in the same allocation. This also covers an `encodings` error from the shared managed Python install. The previous placeholder value for `CUDA_HOME` can be cleared before retrying.

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

Each setup attempt installs its own CPython 3.12.14 under the printed scratch directory. The helper checks the standard library and development headers, prints `Managed Python verified`, and creates the venv from that exact interpreter before installing CUDA or fetching the experiment sources. [Earlier managed Python setup](../../context_pilot_paper/docs/evidencestore/implementation/scheduler_v2_phase1_comparison_scratch_python.sh#L62) · [uv Python setup](https://docs.astral.sh/uv/guides/install-python/)

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
cat "$CHECK_ROOT/python.txt"
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
MODEL="Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION="cdbee75f17c01a7cc42f958dc650907174af0554"
E5_MODEL="intfloat/e5-base-v2"
REPLAY="$REPO/scripts/vllm_pd_replay.py"
export REPO PYTHON RUN_ROOT SMOKE_ROOT SMOKE_TOPOLOGY MODEL MODEL_REVISION E5_MODEL REPLAY
export E5_REVISION="${E5_REVISION:-}"
export PD_LMCACHE_SIZE_GB="${PD_LMCACHE_SIZE_GB:-8}"

if bash <<'SMOKE'
set -euo pipefail
mkdir -p "$SMOKE_ROOT"
E5_REVISION="${E5_REVISION:-$("$PYTHON" -c 'from huggingface_hub import model_info; print(model_info("intfloat/e5-base-v2").sha)')}"
printf '%s\n' "$E5_MODEL@$E5_REVISION" > "$RUN_ROOT/e5-model.txt"
uv pip freeze > "$RUN_ROOT/python-packages.txt"
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
SMOKE
then
  e5_receipt=$(<"$RUN_ROOT/e5-model.txt")
  E5_REVISION="${e5_receipt#*@}"
else
  smoke_status=$?
  printf 'Smoke stopped (exit %s). The allocation shell is available for inspecting %s.\n' \
    "$smoke_status" "$SMOKE_ROOT" >&2
fi
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

For a smoke-only run, publish the passed results below before leaving the allocation. Continue to [Dataset inputs](#dataset-inputs) to run the full datasets in the same job.

### Publish the smoke receipts and traces to Git

After both smoke receipts pass, run this in the same allocation shell. The publisher checks both cohorts, writes a tracked folder under `reports/smoke/`, commits that folder, and pushes `main`. The archive contains the request and timeline reports, raw traces and counters, input and output hashes, and setup provenance. Keep the complete run on scratch or copy it to shared storage using section 3.

```bash
if git -C "$PD_HOME_REPO" pull --ff-only origin main &&
    bash "$PD_HOME_REPO/scripts/vllm_pd_publish_smoke.sh"; then
  printf 'Run archive: reports/smoke/%s\n' "${RUN_ROOT##*/}"
else
  printf 'Publication stopped. Inspect the message above from this allocation shell.\n' >&2
fi
```

After the push, pull `main` in the other checkout. The verifier prints the run ID and checks every archived file's size and SHA-256, along with both cohort receipts.

```bash
git pull --ff-only origin main
for manifest in reports/smoke/*/archive-manifest.json; do
  [[ -f "$manifest" ]] || continue
  python3 scripts/vllm_pd_smoke_archive.py --verify "${manifest%/*}" || break
done
```

### Inspect a failed smoke run

Use the cohort directory printed by the launcher. Look for the first `EngineCore encountered a fatal error.` or `EngineCore failed to start.` and read the traceback that follows. A later `EngineDeadError` records that the API had already lost the engine. The router and workload logs show the requests around the failure.

If the allocation shell is still open, read the logs there. If it has ended and the scratch directory is still present, use the following CPU-only allocation on the same node from the Slurm login shell. Paste the cohort directory ending in `smoke/1p3d-native` or the corresponding failed cohort when prompted. [Earlier same-node closeout](../../context_pilot_paper/docs/evidencestore/implementation/scheduler_v2_phase1_native_1p3d_gpu_run_card.md#L268)

```bash
read -r -p 'Failed smoke cohort directory: ' FAILED_COHORT
srun -p all -A oana2026 -q research12h -N 1 -w gpu-grad-01 \
  -n 1 -c 1 --mem=1G -t 00:05:00 --chdir=/tmp bash -s -- "$FAILED_COHORT" <<'NODE'
logs="$1/logs"
if [[ ! -d "$logs" ]]; then
  printf 'The run logs are unavailable at %s on %s.\n' "$logs" "$(hostname -s)" >&2
  exit 1
fi
for log in "$logs"/{p*,d*,router,workload,metrics-before}.log; do
  [[ -f "$log" ]] || continue
  printf '\n--- %s ---\n' "${log##*/}"
  if ! grep -n -F -m 1 -A 100 -e 'EngineCore encountered a fatal error.' \
      -e 'EngineCore failed to start.' -- "$log"; then
    tail -n 80 -- "$log"
  fi
done
NODE
```

### Retry a tracing fix with the retained runtime

For the NIXL receive tracing fix, the upstream native wheel and dependency pins are the same. After pulling the fix and requesting a new four-GPU allocation on `gpu-grad-01`, you can reuse the previous scratch venv if it is still present. Run the original `source` command printed by runtime setup, then refresh the launcher and bundle below. A failed refresh leaves the shell open. If the scratch runtime is gone or the check fails, build a fresh runtime in section 1.2.

```bash
if bash <<'REFRESH'
set -euo pipefail
test -n "$SLURM_JOB_ID"
test "$(hostname -s)" = gpu-grad-01
old_commit=$(<"$CHECK_ROOT/experiment-source.txt")
new_commit=$(git -C "$PD_HOME_REPO" rev-parse HEAD)
if ! git -C "$PD_HOME_REPO" diff --quiet "$old_commit" "$new_commit" -- . \
    ':(exclude)docs/vllm-native-pd-runbook.md' ':(exclude)scripts/vllm_pd_cluster.py' \
    ':(exclude)third_party/vllm-fork.json' ':(exclude)third_party/vllm-native-pd.bundle'; then
  printf 'The checkout changed beyond the tracing fix. Build a fresh runtime.\n' >&2
  exit 1
fi
receipt_base='import json, sys; print(json.load(sys.stdin)["base_commit"])'
old_base=$("$PYTHON" -c "$receipt_base" < "$REPO/third_party/vllm-fork.json")
new_base=$(git -C "$PD_HOME_REPO" show "$new_commit:third_party/vllm-fork.json" | "$PYTHON" -c "$receipt_base")
test "$old_base" = "$new_base"
git -C "$PD_HOME_REPO" archive "$new_commit" docs/vllm-native-pd-runbook.md \
  scripts/vllm_pd_cluster.py third_party/vllm-fork.json third_party/vllm-native-pd.bundle \
  | tar -xf - -C "$REPO"
"$PYTHON" "$REPO/scripts/vllm_pd_prepare.py"
printf '%s\n' "$new_commit" > "$CHECK_ROOT/experiment-source-refresh-$SLURM_JOB_ID.txt"
REFRESH
then
  printf 'Runtime refreshed. Start a fresh smoke run.\n'
else
  printf 'Refresh failed. Inspect the error above before starting the smoke.\n' >&2
fi
```

### Dataset inputs

The full campaign stages the labeled MuSiQue-Ans v1.0 development file and the corrected April 7 2WikiMultiHopQA development file from their official archives. It also reuses the four prepared MT-RAG Human domains from the smoke. To use already staged files, set `MUSIQUE_DEV` and `TWOWIKI_DEV` to their absolute paths before launching. [MuSiQue data](https://github.com/StonyBrookNLP/musique) · [2WikiMultiHopQA April 7 archive](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip) · [MT-RAG data](https://github.com/IBM/mt-rag-benchmark)

| Evaluation input | Multi-turn conversations | Requests per full cohort |
| --- | ---: | ---: |
| MuSiQue-Ans v1.0 dev | Single-turn questions | 2,417 |
| 2WikiMultiHopQA April 7 dev | Single-turn questions | 12,576 |
| MT-RAG Human, ClapNQ | 29 | 224 |
| MT-RAG Human, Cloud | 26 | 205 |
| MT-RAG Human, FiQA | 27 | 199 |
| MT-RAG Human, Government | 28 | 214 |
| **Total** | **110 multi-turn conversations** | **15,835** |

The two QA counts come from their published dev splits. The MT-RAG counts come from unique conversation and task IDs in the pinned Human generation file. Four full cohorts give **63,340 dataset requests**, plus the controlled probes. Preparation checks the supplied files before indexing, and replay receipts verify how many completed. [MuSiQue-Ans Table 2](https://aclanthology.org/2022.tacl-1.31.pdf#page=8) · [2WikiMultiHopQA Table 1](https://aclanthology.org/2020.coling-main.580.pdf#page=6) · [Pinned MT-RAG Human tasks](https://github.com/IBM/mt-rag-benchmark/blob/e0970de1b1207c2a9506adf238e6e1a22037e38a/mtrag-human/generation_tasks/reference.jsonl)

The corrected 2Wiki dev file contains 70 out-of-range supporting sentence references. Preparation retains all questions, paragraphs, and title-level support, and records the unresolved sentence references in `data/2wiki/manifest.json`.

The indexes use the allocated GPU for corpus embedding before the serving workers start. The full campaign reads the embedding model revision saved by the smoke. The live retrieval run embeds queries on the CPU, records search stages, and freezes the ranked IDs and resulting tokenized prompts. Paired runs read those same prompt IDs. `dispatch_to_first_token_ms` measures serving after dispatch. For a throughput or latency claim, run each arm with the same frozen input and concurrency so they share a comparable CPU and arrival load. The live capture also reports latency from before embedding.

## 2. Run both layouts and all datasets

The launcher starts four independent workers, validates the physical GPUs and editable fork, captures native counters before and after the workload, and tears down the process groups. `--dry-run` prints the effective settings. The native full-handoff reference runs with `--prefix-cache off`. The shared arm runs with `--lmcache --lmcache-on-decode --prefix-cache on`. Each cohort starts with fresh worker and server caches. To measure the prefill-only LMCache control in isolation, use `--lmcache --prefix-cache off`.

Run this from the active allocation shell that completed both smoke arms. Keep the shell open and detach with `Ctrl+b` then `d` if you are using tmux. The script stages the inputs and builds all six Faiss indexes before serving. It then runs the controlled probes and all datasets in each cohort. A failure leaves the allocation shell open.

```bash
if git -C "$PD_HOME_REPO" pull --ff-only origin main &&
    bash "$PD_HOME_REPO/scripts/vllm_pd_full_campaign.sh"; then
  printf 'Full measurement passed. Results: %s\n' "$RUN_ROOT"
fi
```

Each cohort writes its request, transfer, and timeline reports below `$RUN_ROOT`. During serving, inspect `logs/workload.log` under the cohort directory printed by the launcher for per-dataset progress. Save the full results to shared storage using [section 3](#3-check-the-receipts-in-the-allocation).

### Review each dataset as it finishes

Each cohort runs MuSiQue, 2Wiki, and the four MT-RAG domains in sequence. After each slice, it verifies the replay count and checks the frozen prompts against the first cohort where applicable. It joins that slice's trace events and reconciles completed NIXL bytes with native counter snapshots taken immediately before and after it. The report includes serving latency, GPU batch timing, scheduled and reused evidence tokens for P and D, and the NIXL byte attribution. Detailed request, document, timeline, and counter files are under `<cohort>/datasets/<label>/` in the run directory.

Each passing summary is committed and pushed to `reports/datasets/<run-id>/<label>/<cohort>.json`. A standard campaign produces 24 summaries. Use the shared checkout's SSH push URL from the smoke publication. If publication fails, the campaign retains the report and continues. The next publication includes queued commits, and the driver retries pending reports after all cohorts finish. Inspect `logs/workload.log` for `Published` or `Git publication pending`.

To retry any remaining pending reports from the active allocation shell, run the following.

```bash
for pending in "$RUN_ROOT"/{1p3d,2p2d}-{native,shared}/datasets/*/summary.json.pending; do
  [[ -f "$pending" ]] || continue
  if "$PYTHON" "$PD_HOME_REPO/scripts/vllm_pd_publish_dataset.py" --report "${pending%.pending}"; then
    rm -- "$pending"
  fi
done
```

If the allocation has ended, use `python3 scripts/vllm_pd_publish_dataset.py --retry <run-id>` from the shared checkout to push any existing checkpoint commits. If measurement stopped partway, the completed dataset summaries remain available. Start a fresh run directory before relaunching measurement.

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

### Publish the detailed campaign measurements

After the four cohort summaries for a dataset are pushed, publish that dataset's detailed review files from the allocation shell. The publisher uses the shared checkout's SSH push URL.

```bash
if git -C "$PD_HOME_REPO" pull --ff-only origin main &&
    "$PYTHON" "$PD_HOME_REPO/scripts/vllm_pd_publish_dataset.py" \
      --details "$RUN_ROOT" --dataset mtrag/fiqa; then
  printf 'Detailed FiQA results pushed.\n'
else
  printf 'Publication stopped. The allocation shell is still open.\n' >&2
fi
```

Repeat with `musique`, `2wiki`, `mtrag/clapnq`, `mtrag/cloud`, `mtrag/fiqa`, or `mtrag/govt`. Each dataset lands under `reports/datasets/<run-id>/details/<label>/` in one commit. Its manifest lists file hashes and request and document counts across the four cohorts. The archive includes individual receipt metadata, joined request measurements, document and chunk measurements, and sampled timelines. Read `.jsonl.gz` files with `gzip -dc`. Keep the original streaming payloads, frozen prompts, corpus indexes, and raw engine events in the shared-storage copy above.

To publish from the shared-storage copy later, pass `"$archive_dir/run"` to `--details`. Omit `--dataset` to publish all six datasets after all four cohorts complete. If a push fails after the archive commit, retry with `python3 scripts/vllm_pd_publish_dataset.py --retry <run-id>` from the shared checkout.

## Measure the same build with tracing on and off

The same editable fork serves both runs. `--trace on` sets `EVIDENCESTORE_PD_TRACE=1` on each worker and enables request, scheduler, GPU batch, and NIXL completion events. `EVIDENCESTORE_TRACE_DIR` supplies their output directory. `--trace off` sets the source gate to `0`. The replay and router still capture client-side timings, and the reporter still reads the native aggregate NIXL counters. These runs report per-request engine work and payload as unavailable. `EVIDENCESTORE_TRACE_GPU=0` skips CUDA event timing within a trace-on run.

For an overhead check, replay the same frozen fixture in the same layout and prefix mode into two fresh run directories. Set `--trace on` for one and `--trace off` for the other. Match the source commit, model, GPU placement, and replay manifest, then compare client timing and native counter deltas. The earlier repo used the same runtime-gate pattern for its timing experiments. [Earlier measurement gate](../../context_pilot_paper/scripts/v4_es_matrix.sh#L46)

For a source comparison, commit each revision directly in `third_party/forks/vllm/` and run the same replay with the same trace settings for both builds. Use `python3 scripts/vllm_pd_prepare.py --export` to refresh the bundle and source receipt when transferring a revision to a different machine.
