# Experiment scripts

`checkout_sources.py` materializes the candidate source commits from `third_party/sources.json`. Run `python3 scripts/checkout_sources.py --verify` to verify every local checkout, or append source names to select specific repositories.

| Script | Purpose |
| --- | --- |
| `vllm_pd_prepare.py` | Materialize the committed vLLM fork or export a new committed revision with `--export`. |
| `vllm_pd_runtime.sh` | Stage the pinned experiment tree and build the editable runtime in node-local Slurm scratch. |
| `vllm_pd_cluster.py` | Launch 1P/3D or 2P/2D in an existing allocation and write the effective settings. |
| `vllm_pd_replay.py` | Prepare and index the full data, run controlled cases, and replay live or frozen Faiss results. |
| `vllm_pd_report.py` | Capture native NIXL counters and generate the joined request timelines and report. |
| `vllm_pd_publish_smoke.sh` | Archive and push a passed smoke pair from the shared checkout. |
| `vllm_pd_full_campaign.sh` | Stage and index all dataset inputs and replay the four cohorts in the current allocation. |
| `vllm_pd_legacy_exponent_replay.py` | Replay a verified KV page campaign on CPU with standalone and forward-pinned exponent-delta storage at matched cache budgets. |
| `vllm_pd_mtrag_scale_select.py` | Freeze the held-out MT-RAG conversation queue from the verified four-domain retrieval traces. |
| `vllm_pd_mtrag_scale_run.py` | Run one conversation at a time through GPU capture, CPU replay, publication, and verified scratch cleanup. |
| `vllm_pd_mtrag_scale_publish.py` | Publish exact raw archives as GitHub Release assets or split Git blobs, checkpoint CPU reports, and verify or download a checkpoint. |
| `vllm_pd_chunk_smoke.sh` | Run the MuSiQue chunk-trace smoke or use `--full` for one pass in `2p2d-shared`. Add `--paired` for a native comparison. |
| `vllm_pd_blend_smoke.py` | Test controlled prefix-to-Blend reuse or use `--musique-prepared` to test natural Blend reuse in both four-GPU layouts. |
| `vllm_pd_blend_natural_report.py` | Verify live MuSiQue replay and attribute completed shifted LMCache restores to repeated evidence. |
| `vllm_pd_musique_repeat_report.py` | Verify the paired live inputs and compare naturally repeated chunks across cohorts. |
| `vllm_pd_archive_musique.py` | Package the full MuSiQue chunk and transfer measurements for review. |
| `vllm_pd_dataset_report.py` | Capture dataset counter boundaries and report joined request and document measurements. |
| `vllm_pd_publish_dataset.py` | Commit and push verified dataset summaries or the detailed campaign review archive, and retry pending pushes. |
| `vllm_pd_stage_inputs.py` | Download and verify the official labeled development files used by the full campaign. |
| `vllm_pd_smoke_archive.py` | Verify a pulled smoke archive against its file hashes and cohort receipts. |

The [native PD runbook](../docs/vllm-native-pd-runbook.md) gives the commands for the entire Slurm allocation.

The [MT-RAG scale runbook](../docs/mtrag-kv-capture-scale-plan.md#run-the-pilot-and-campaign) gives the commands for the raw-page pilot, larger capture, and independent CPU replay.
