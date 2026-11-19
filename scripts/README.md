# Experiment scripts

`checkout_sources.py` materializes the candidate source commits from `third_party/sources.json`. Run `python3 scripts/checkout_sources.py --verify` to verify every local checkout, or append source names to select specific repositories.

| Script | Purpose |
| --- | --- |
| `vllm_pd_prepare.py` | Materialize the committed vLLM fork or export a new committed revision with `--export`. |
| `vllm_pd_cluster.py` | Launch 1P/3D or 2P/2D in an existing allocation and write the effective settings. |
| `vllm_pd_replay.py` | Prepare and index the full data, run controlled cases, and replay live or frozen Faiss results. |
| `vllm_pd_report.py` | Capture native NIXL counters and generate the joined request timelines and report. |

The [native PD runbook](../docs/vllm-native-pd-runbook.md) gives the commands for the entire Slurm allocation.
