# EvidenceStore PD measurements

Measure prefill and decode GPU work, KV bytes by path, and serving latency in **1P/3D** and **2P/2D** on the same four GPUs. Compare full native NIXL handoffs with a shared LMCache server serving P and D and native D prefix reuse enabled. Later comparisons use the same Faiss replay and accounting for CacheBlend, SGLang, Mooncake, PPD, and EvidenceStore.

| Directory | Contents |
| --- | --- |
| [`docs/`](docs/) | Slide 4 plan and instrumentation requirements for each engine and backend. |
| [`scripts/`](scripts/) | Source preparation, native PD launch, dataset replay, and reporting commands. |
| [`src/`](src/) | Project-owned gateway, telemetry, replay, and baseline adapters. |
| [`reports/`](reports/) | Prior-run review and experiment reports. Generated run artifacts live in `reports/runs/`. |
| [`third_party/`](third_party/) | Upstream pins, detached seed checkouts, and the directly editable local vLLM fork. |

- [Slide 4 measurement plan](docs/slide-4-pd-measurement-plan.md)
- [Native vLLM PD runbook](docs/vllm-native-pd-runbook.md)
- [Request-level PD instrumentation](docs/pd-instrumentation.md)
- [Earlier PD runs and reusable evidence](reports/prior-pd-review.md)
- [Pinned upstream sources](third_party/README.md)

From the repo root, `python3 scripts/checkout_sources.py --verify` checks every local source against the manifest without fetching. `python3 scripts/checkout_sources.py` materializes any missing checkout.

`python3 scripts/vllm_pd_prepare.py` materializes the committed vLLM fork at `third_party/forks/vllm/`. Edit the fork's source directly. The same editable build runs with native tracing on or off.
