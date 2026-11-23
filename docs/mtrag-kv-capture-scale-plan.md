# Scale the MT-RAG KV capture

Job 30568 captured raw complete P-side KV pages from six CLAPNQ conversations. The CPU replay compares standalone compression with exact exponent deltas on those saved pages. Complete the six-conversation replay as the regression check, then validate on **20 new conversations**, five from each MT-RAG domain. Use one frozen selection and extend it in stages if the early results justify the capture cost. [Original six captures][campaign] · [CPU replay][replay]

## Conversation coverage

The pinned MT-RAG Human workload has **110 conversations and 842 turns**. The six original CLAPNQ conversations were selected for high reuse. Keep those as regression cases while selecting new conversations over a range of turn lengths and repeat rates. Choose the cases before looking at compression and hit-rate results. Capture each conversation from turn 1 through its final turn so later page loads can be matched to earlier stores. [Dataset counts][dataset] · [Original selection][selection]

| Domain | Conversations | Turns | New conversations in the first batch |
| --- | ---: | ---: | ---: |
| CLAPNQ | 29 | 224 | 5 |
| Cloud | 26 | 205 | 5 |
| FiQA | 27 | 199 | 5 |
| Govt | 28 | 214 | 5 |
| **Total** | **110** | **842** | **20** |

The frozen queue records each domain, conversation ID, and full turn window. The full run adds the remaining cases for **104 new plus six already captured**. The queue and progress survive a Slurm job ID change.

Use the same 1B/3D topology, docs-first `previous-turn-first` policy, fixed reference history, and native NIXL handoff for the capture. Reuse one verified runtime and the prepared Faiss index for each domain. Start each conversation with a fresh worker cache. The scale path escapes literal `¶` characters in source text so the actual Blend separators remain distinct. The report records that setting and any difference between live and archived retrieval ranks. If a conversation reaches a page or byte cap, retry it with enough capture budget. [Original capture runbook][capture]

## Capture, publish, then reclaim scratch

The scale runner finishes the entire lifecycle of one conversation before starting another.

1. **Capture one conversation.** Close its original archive and verify natural turn coverage, the captured tensor digests, store and load events, and native movement reconciliation.
2. **Replay its saved bytes.** Compare standalone Zstd field frames with the exponent-delta graph using the same memory budgets, fresh logical caches, and recorded chronology. Write a per-conversation CPU receipt. Include zero-reuse cases in the final report.
3. **Archive the payload.** Upload the original archive to a durable destination. Preserve its SHA-256, byte count, and any ordered part list. Verify the durable bytes against the original digest. Verify all parts when splitting an archive.
4. **Publish the review checkpoint.** Push the lightweight Git record with the conversation and turn window, raw archive location, digest, capture provenance, and CPU results. Check that the remote contains the committed receipt. Record success only after both the raw artifact and its receipt are verified.
5. **Reclaim this conversation's scratch.** Remove the exact completed local run tree, local archive, and temporary publication checkout once their verified remote records are ready. Keep the runtime and prepared indexes for the next conversation.

If capture or publication fails, retain that conversation's scratch and retry from its last verified step. On restart, use the persisted per-conversation receipts to skip already verified cases, including those whose local archives were cleaned. A retry that finds a matching remote archive should verify it and publish the missing receipt. A retry after receipt publication can finish the cleanup. A lost scratch capture can be rerun as the same plan entry with a new attempt receipt. Avoid retaining downloaded tensor objects in a long-lived Git checkout.

## Where the data lives

The six published packets occupy about **5.18 GiB**. Scaling their average is useful for planning, with turn count and KV usage deciding the actual volume. [Published packet inventory][campaign]

| Conversations captured | Approximate packet volume at the original average |
| ---: | ---: |
| 20 | 17 GiB |
| 40 | 35 GiB |
| 60 | 52 GiB |
| 110 | 95 GiB |

For this volume, Git should hold the manifests, checksums, and reports. Store the original archives in durable shared project storage or as GitHub Release assets if keeping them in the same GitHub project is preferred. Release assets are limited to less than 2 GiB each, so split larger archives into ordered parts. GitHub recommends keeping the Git object store within 10 GB and caps each Git push at 2 GiB. Using a new Git branch for each capture still adds bytes to the repository's object store. [GitHub repository limits][github-limits] · [GitHub Release asset limits][github-assets]

## Run the one-conversation pilot over SSH

Use this path from an existing four-GPU allocation on `gpu-grad-01` when GitHub SSH push works. It publishes the exact archive in parts to `results/mtrag-scale-01` and reads it back before deleting the local capture. The archive and CPU report are publicly readable. This path needs Git, `curl`, and the GitHub SSH key. Use the Release workflow below for a larger campaign to keep the raw payloads outside the Git object store. [GitHub repository limits][github-limits]

```bash
(
  set -euo pipefail
  umask 077
  : "${SLURM_JOB_ID:?Run inside the existing four-GPU allocation}"
  test "$(hostname -s)" = gpu-grad-01
  campaign=mtrag-scale-01
  github_repo=Kibnelson/evidence_store_1
  git_url="https://github.com/$github_repo.git"
  branch="refs/heads/results/$campaign"
  node_root="/tmp/$(id -un)"
  mkdir -p "$node_root"
  [[ -O "$node_root" && ! -L "$node_root" ]]
  df -h "$node_root"

  remote_branch=$(git ls-remote --refs "$git_url" "$branch")
  if [[ -n "$remote_branch" ]]; then
    branch_commit=${remote_branch%%$'\t'*}
    source_commit=$(curl -fsSL --retry 3 \
      "https://raw.githubusercontent.com/$github_repo/$branch_commit/reports/campaigns/$campaign/plan.json" \
      | python3 -c 'import json, sys
p = json.load(sys.stdin)
if p.get("archive_backend", "release") != "git-ssh":
    raise SystemExit("Existing campaign uses Release. Resume it with Release or choose a new campaign ID.")
print(p["source_commit"])')
  else
    main_ref=$(git ls-remote --exit-code --refs "$git_url" refs/heads/main)
    source_commit=${main_ref%%$'\t'*}
  fi
  [[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]

  source_repo="$node_root/mtrag-scale-source-$campaign"
  plan="$node_root/$campaign.plan.json"
  if [[ ! -d "$source_repo/.git" ]]; then
    git init -q "$source_repo"
  fi
  if ! git -C "$source_repo" cat-file -e "$source_commit^{commit}" 2>/dev/null; then
    git -C "$source_repo" fetch --depth=1 --no-tags "$git_url" "$source_commit"
  fi
  git -C "$source_repo" checkout --detach "$source_commit"
  cd "$source_repo"
  python3 -B scripts/vllm_pd_mtrag_scale_select.py --output "$plan"
  export EVIDENCESTORE_LEGACY_CUDA_HOME="${EVIDENCESTORE_LEGACY_CUDA_HOME:-/usr/local/pkgs/cuda/12.6}"
  python3 -B scripts/vllm_pd_mtrag_scale_run.py \
    --campaign "$campaign" --plan "$plan" --through 1 --read-back \
    --archive-backend git-ssh 2>&1 | tee -a "$node_root/$campaign.log"
)
```

Look for `Verified published checkpoint` and `Scratch reclaimed`. Rerun the same block to verify the existing checkpoint if the pilot already completed, or resume from the retained local archive after a publication failure. The source comes from the published campaign plan whenever the branch exists. To start a separate Release campaign later, use another campaign ID such as `mtrag-scale-02` in all the Release commands.

## Run the six cumulative milestones

Start the milestone launcher inside a four-GPU allocation after the one-conversation pilot has published the campaign plan. These counts are cumulative over one frozen queue. A milestone is ready for review when its JSON appears on the results branch. Each conversation has its own CPU report and verified archive checkpoint before its local run tree and archive are removed. The runtime and domain indexes are reused for the next conversation.

| Milestone | Cumulative conversations | Cumulative turns | Review file |
| ---: | ---: | ---: | --- |
| 1 | 1 | 7 | `milestones/001.json` |
| 20 | 20 | 155 | `milestones/020.json` |
| 40 | 40 | 307 | `milestones/040.json` |
| 60 | 60 | 458 | `milestones/060.json` |
| 80 | 80 | 593 | `milestones/080.json` |
| 104 | 104 | 802 | `milestones/104.json` |

The launcher fetches the source commit pinned by the existing plan and uses its recorded archive destination. For `mtrag-scale-01`, that means all raw archives go into Git history over SSH. At the pilot's 1.36 GiB per archive, 104 would add about **141 GiB**. GitHub recommends a 10 GB Git object store. A Release-backed campaign uses Git for the same reports and receipts and keeps raw archives in Release assets. Start its pilot with a separate ID such as `mtrag-scale-02`, then set that ID in the launcher command below. [GitHub repository limits][github-limits] · [GitHub Release asset limits][github-assets]

```bash
(
  set -euo pipefail
  umask 077
  : "${SLURM_JOB_ID:?Run inside the four-GPU allocation}"
  campaign=mtrag-scale-01
  node_root="/tmp/$(id -un)"
  mkdir -p "$node_root"
  [[ -O "$node_root" && ! -L "$node_root" ]]
  control="$node_root/mtrag-scale-control"
  if [[ ! -d "$control/.git" ]]; then
    git clone --depth=1 --single-branch --branch main --no-tags --filter=blob:none --sparse \
      https://github.com/Kibnelson/evidence_store_1.git "$control"
  else
    git -C "$control" pull --ff-only origin main
  fi
  git -C "$control" sparse-checkout set --no-cone '/scripts/vllm_pd_mtrag_scale_*'
  export EVIDENCESTORE_LEGACY_CUDA_HOME="${EVIDENCESTORE_LEGACY_CUDA_HOME:-/usr/local/pkgs/cuda/12.6}"
  python3 -B "$control/scripts/vllm_pd_mtrag_scale_stages.py" \
    --campaign "$campaign" --through 104 2>&1 | tee -a "$node_root/$campaign.log"
)
```

Change `--through 104` to `--through 20` to stop at the first balanced checkpoint. Append `--read-back` to download and hash each new archive in addition to checking the committed remote part identities. Keep tmux open or detached while the job runs. If Slurm ends the allocation or a push fails, rerun the same block in an active four-GPU allocation. The launcher skips published milestones and the capture runner resumes the first incomplete stage from its per-conversation receipts.

The [SSH milestone reports][ssh-milestones] summarize the domain and turn counts, archived bytes, and CPU cache hits and misses at shared budgets. Each report links to the exact conversation receipts and CPU studies used for that stage.

### Review the SSH pilot on CPU

Run from a source checkout with the SSH publisher. The commands fetch the plan and one archive by conversation and rerun the exact CPU replay. The published report is in `reports/campaigns/mtrag-scale-01/studies/` on the results branch.

```bash
(
  set -euo pipefail
  umask 077
  campaign=mtrag-scale-01
  cpu_scratch="${EVIDENCESTORE_CPU_SCRATCH:-${TMPDIR:-/tmp}}"
  review=$(mktemp -d "$cpu_scratch/mtrag-cpu-$campaign.XXXXXX")
  publisher=(python3 -B scripts/vllm_pd_mtrag_scale_publish.py \
    --campaign "$campaign" --archive-backend git-ssh --scratch-dir "$review")
  "${publisher[@]}" plan > "$review/plan.json"
  conversation=$(python3 -c 'import json, sys; print(json.load(sys.stdin)["plan"]["conversations"][0]["conversation_id"])' \
    < "$review/plan.json")
  "${publisher[@]}" status --conversation "$conversation" > "$review/checkpoint.json"
  read -r archive_name archive_sha256 turn_window < <(python3 -c \
    'import json, sys; r=json.load(sys.stdin); print(r["archive"]["filename"], r["archive"]["sha256"], r["turn_window"])' \
    < "$review/checkpoint.json")
  "${publisher[@]}" download --conversation "$conversation" --output "$review/$archive_name"
  python3 -B scripts/vllm_pd_legacy_exponent_replay.py \
    --archive "$review/$archive_name" --sha256 "$archive_sha256" \
    --conversation "$conversation" --turn-window "$turn_window" \
    --scratch-dir "$review" --output-dir "$review/cpu" \
    --capacity-mib 96 128 160 192 224 256 384 512 768 1024 1536 2048 3072
  printf 'CPU report %s\n' "$review/cpu/$conversation.json"
)
```

## Run a campaign with GitHub Releases

The commands use the public `Kibnelson/evidence_store_1` repository. Each raw archive goes to GitHub Release `mtrag-scale-02`. The plan, capture receipts, and CPU study reports go to branch `results/mtrag-scale-02`. The release assets are publicly readable. Use the same campaign ID until all intended stages of its frozen queue finish. [Release asset verification][github-api]

<a id="run-the-pilot-and-campaign"></a>

### 1. Start tmux and request the allocation

Start on the login node once the scale-runner files are on GitHub `main`. The GPU node fetches that exact commit into node scratch. For a resumed campaign, the command reads its original source commit from the published plan. The publisher uses authenticated GitHub CLI for Release assets and your existing GitHub SSH key for Git pushes. If `gh auth status` fails, sign in with `gh auth login --hostname github.com --web` and rerun the block. Python 3.10 or later, a C compiler, and `libzstd` are required for CPU replay. [GitHub CLI installation][gh-install] · [GitHub CLI authentication][gh-auth]

```bash
(
  set -euo pipefail
  test -z "${SLURM_JOB_ID:-}"
  campaign=mtrag-scale-02
  github_repo=Kibnelson/evidence_store_1
  git_url="https://github.com/$github_repo.git"

  gh auth status --hostname github.com
  branch=$(git ls-remote --heads "$git_url" "refs/heads/results/$campaign")
  if [[ -n "$branch" ]]; then
    source_commit=$(gh api -H 'Accept: application/vnd.github.raw+json' \
      "repos/$github_repo/contents/reports/campaigns/$campaign/plan.json?ref=results/$campaign" \
      | python3 -c 'import json, sys; print(json.load(sys.stdin)["source_commit"])')
  else
    main_ref=$(git ls-remote --exit-code --refs "$git_url" refs/heads/main)
    source_commit=${main_ref%%$'\t'*}
    gh api --silent "repos/$github_repo/contents/scripts/vllm_pd_mtrag_scale_run.py?ref=$source_commit"
  fi
  [[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]

  module load slurm
  srun_bin=$(command -v srun)
  "$srun_bin" --version
  tmux -L evidence-store-mtrag-scale -f /dev/null new-session -s mtrag-scale \
    -e "PATH=$PATH" -e "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" \
    -e "SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-}" \
    -e "MTRAG_SCALE_CAMPAIGN=$campaign" -e "MTRAG_SCALE_SOURCE=$source_commit" \
    "$srun_bin" -p all -A oana2026 -q research12h -N 1 -w gpu-grad-01 \
      --gpus=4 -c 16 --mem=192G -t 12:00:00 --pty /bin/bash -l
)
```

The tmux command hands `srun` its resolved executable path from the login shell. Detach with `Ctrl+b` then `d`. Reattach from the same login node.

```bash
tmux -L evidence-store-mtrag-scale attach-session -t mtrag-scale
```

### 2. Prepare the source and preview the queue

Paste this inside the allocated GPU shell. It prepares a detached source checkout and a frozen queue in node scratch. Selectors use the archived request history to span turn lengths and repeat rates within each domain. The capture still runs live Faiss retrieval against the pinned corpus. [Archived retrieval selection][selection]

```bash
(
  set -euo pipefail
  umask 077
  : "${SLURM_JOB_ID:?Run this inside the four-GPU allocation}"
  : "${MTRAG_SCALE_CAMPAIGN:?Start the allocation from step 1}"
  : "${MTRAG_SCALE_SOURCE:?Start the allocation from step 1}"
  gh auth status --hostname github.com

  node_root="/tmp/$(id -un)"
  mkdir -p "$node_root"
  [[ -O "$node_root" && ! -L "$node_root" ]]
  df -h "$node_root"
  source_repo="$node_root/mtrag-scale-source-$MTRAG_SCALE_CAMPAIGN"
  plan="$node_root/$MTRAG_SCALE_CAMPAIGN.plan.json"

  if [[ ! -d "$source_repo/.git" ]]; then
    git init -q "$source_repo"
    git -C "$source_repo" remote add origin https://github.com/Kibnelson/evidence_store_1.git
  fi
  if ! git -C "$source_repo" cat-file -e "$MTRAG_SCALE_SOURCE^{commit}" 2>/dev/null; then
    git -C "$source_repo" fetch --depth=1 --no-tags origin "$MTRAG_SCALE_SOURCE"
  fi
  git -C "$source_repo" checkout --detach "$MTRAG_SCALE_SOURCE"
  cd "$source_repo"
  if [[ ! -s "$plan" ]]; then
    python3 -B scripts/vllm_pd_mtrag_scale_select.py --output "$plan"
  fi
  python3 -B scripts/vllm_pd_mtrag_scale_run.py \
    --campaign "$MTRAG_SCALE_CAMPAIGN" --plan "$plan" --through 20 --dry-run
)
```

### 3. Run one conversation, then extend to 20

Start with `through=1` and read back the uploaded archive. The replay runs on CPU after the serving workers exit and publishes its report alongside the capture receipt. Watch for `Verified published checkpoint` and `Scratch reclaimed`. Rerunning the pilot should skip the capture and verify the existing checkpoint. Once the pilot passes, change `through` to `20` and set `read_back=no` to check subsequent uploads using GitHub's SHA-256 for each uploaded part. [Release asset API][github-api]

```bash
through=1
read_back=yes
(
  set -euo pipefail
  umask 077
  : "${SLURM_JOB_ID:?Run this inside the four-GPU allocation}"
  : "${MTRAG_SCALE_CAMPAIGN:?Start the allocation from step 1}"
  node_root="/tmp/$(id -un)"
  cd "$node_root/mtrag-scale-source-$MTRAG_SCALE_CAMPAIGN"
  export EVIDENCESTORE_LEGACY_CUDA_HOME="${EVIDENCESTORE_LEGACY_CUDA_HOME:-/usr/local/pkgs/cuda/12.6}"
  args=()
  if [[ "$read_back" == yes ]]; then args+=(--read-back); fi
  python3 -B scripts/vllm_pd_mtrag_scale_run.py \
    --campaign "$MTRAG_SCALE_CAMPAIGN" \
    --plan "$node_root/$MTRAG_SCALE_CAMPAIGN.plan.json" \
    --through "$through" "${args[@]}" 2>&1 | tee -a "$node_root/$MTRAG_SCALE_CAMPAIGN.log"
)
```

Use the same command with `through=40`, `through=60`, or `through=104` for later stages. The number is the last position in the frozen queue. Published entries still count toward that number. Run one campaign process at a time. If a GPU allocation ends, repeat steps 1–3 with the same campaign ID. A complete local archive survives an upload failure on the same node. If it disappears before publication, the runner captures that conversation again. A capture that hits the default 6 GiB raw page byte limit can be retried by adding `--capture-mib 8192` if scratch has room. The runtime, indexes, current trace tree, archive, and any upload part share the node filesystem. Set `--keep-local` on the runner if you need the pilot's local run tree for inspection.

To monitor from another login window, get the running job ID from `squeue --me`, then attach a lightweight step to the same GPU node. Enter the job ID at the prompt.

```bash
module load slurm
squeue --me
read -r -p 'Running Slurm job ID: ' capture_job
srun --jobid="$capture_job" --overlap --gres=none -N 1 -n 1 -c 1 \
  bash -c 'tail -n 35 -F "/tmp/$(id -un)/mtrag-scale-02.log"'
```

### 4. Download and replay the pilot on CPU

Run this from the root of a source checkout containing the scale scripts, on a workstation or in a CPU-only allocation. The block reads the first conversation from the published plan, downloads and verifies its original archive, and checks reconstruction from those bytes. Set `conversation` to another published ID to review a different case. Use `EVIDENCESTORE_CPU_SCRATCH` to choose a volume with room for the downloaded archive and the temporary extracted pages. [Original six-conversation CPU replay][replay]

```bash
(
  set -euo pipefail
  umask 077
  campaign=mtrag-scale-02
  github_repo=Kibnelson/evidence_store_1
  test -r scripts/vllm_pd_mtrag_scale_publish.py
  cpu_scratch="${EVIDENCESTORE_CPU_SCRATCH:-${TMPDIR:-/tmp}}"
  review=$(mktemp -d "$cpu_scratch/mtrag-cpu-$campaign.XXXXXX")
  printf 'CPU review directory %s\n' "$review"
  gh api -H 'Accept: application/vnd.github.raw+json' \
    "repos/$github_repo/contents/reports/campaigns/$campaign/plan.json?ref=results/$campaign" \
    > "$review/plan.json"
  conversation=$(python3 -c 'import json, sys; print(json.load(sys.stdin)["plan"]["conversations"][0]["conversation_id"])' \
    < "$review/plan.json")
  publisher=(python3 -B scripts/vllm_pd_mtrag_scale_publish.py --campaign "$campaign" --scratch-dir "$review")
  "${publisher[@]}" status --conversation "$conversation" > "$review/checkpoint.json"
  read -r archive_name archive_sha256 turn_window < <(python3 -c \
    'import json, sys; r=json.load(sys.stdin); print(r["archive"]["filename"], r["archive"]["sha256"], r["turn_window"])' \
    < "$review/checkpoint.json")
  "${publisher[@]}" download --conversation "$conversation" --output "$review/$archive_name"
  python3 -B scripts/vllm_pd_legacy_exponent_replay.py \
    --archive "$review/$archive_name" --sha256 "$archive_sha256" \
    --conversation "$conversation" --turn-window "$turn_window" \
    --scratch-dir "$review" --output-dir "$review/cpu" \
    --capacity-mib 96 128 160 192 224 256 384 512 768 1024 1536 2048 3072
  printf 'CPU report %s\n' "$review/cpu/$conversation.json"
)
```

`scope` in the CPU report counts stored pages, later loads, and tested base pairs. `capacity_results` gives hits, misses, target reads, cold base reads, and rewrites at each memory budget. `matched_cpu_times` gives timings on that CPU for standalone restore, restore with an already decoded base, and restore with an encoded base. Compare timings collected on the same host. The original per-conversation CPU reports are under `reports/campaigns/mtrag-scale-02/studies/` on the results branch.

## What the bigger run can establish

The offline comparison can show how often valid deltas beat standalone compression, whether that saves enough host capacity to retain useful pages, and what the cold and decoded-base restores cost. Report per-conversation and per-domain results alongside totals, and preserve the distinction between the six repeat-heavy regression cases and the new cases. Compare exactness, local hits, cold base bytes, rewrites, and CPU time under the same budgets.

The first two byte-level replays already show why conversation coverage matters. With a 128 MiB cache, forward pinning raised local page hits from 207 to 213 in `1534a095` and lowered them from 115 to 105 in `c1e53df9`. Every hit at each policy's first full-hit budget reproduced the captured bytes. [Five-turn capture][five-turn] · [Six-turn capture][six-turn]

For serving performance, run a separate gated live test that actually stores encoded frames in LMCache. Compare raw, standalone compression, and differential storage on matched requests while keeping the P → D handoff fixed. Measure peak host RAM including scratch, prefill fallback work, and time to first token. This distinguishes the design supported by the capture and CPU replay from its effect on the live serving path. [Design reference][design]

[campaign]: https://github.com/Kibnelson/evidence_store_1/blob/a7b976f/reports/campaigns/clapnq-30568.json
[replay]: vllm-legacy-full-blend-runbook.md#replay-exponent-deltas-using-the-published-captures
[dataset]: datasets/mtrag.md#workload-and-repeat-coverage
[selection]: datasets/repeat-workloads.md
[capture]: vllm-legacy-full-blend-runbook.md#capture-the-six-complete-conversations-in-one-allocation
[design]: kv-differential-storage-reference.md
[five-turn]: https://github.com/Kibnelson/evidence_store_1/tree/a7b976f/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30568-20261123T021621Z
[six-turn]: https://github.com/Kibnelson/evidence_store_1/tree/a7b976f/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30568-20261123T022559Z
[github-limits]: https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits
[github-assets]: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas
[github-api]: https://docs.github.com/en/rest/releases/assets
[ssh-milestones]: https://github.com/Kibnelson/evidence_store_1/tree/results/mtrag-scale-01/reports/campaigns/mtrag-scale-01/milestones
[gh-auth]: https://cli.github.com/manual/gh_auth_status
[gh-install]: https://github.com/cli/cli#installation
