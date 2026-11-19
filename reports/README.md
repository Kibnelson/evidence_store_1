# Reports and run artifacts

Start with the [earlier PD run review](prior-pd-review.md) for the existing experiment evidence and the instrumentation gaps it identifies.

Each new run gets a directory at `reports/runs/<run-id>/` with the source and topology manifest, replay receipt, raw JSONL events, profiler and hardware-counter artifacts, and summary. Preserve the raw event and output-quality receipts when publishing a summary so each reported time and byte total can be checked. Version the summaries and small receipts used to support comparisons.

Passed FiQA smoke pairs are published under `reports/smoke/<run-id>/`. The archive records file hashes, native and shared cohort checks, model and source provenance, counter snapshots, and request traces. [Publish and verify a smoke archive](../docs/vllm-native-pd-runbook.md#publish-the-smoke-receipts-and-traces-to-git)

Full-run dataset summaries are published under `reports/datasets/<run-id>/<label>/<cohort>.json` as each cohort finishes a dataset. The raw traces and detailed document reports are collected in the corresponding scratch run for durable archival. [Review each dataset as it finishes](../docs/vllm-native-pd-runbook.md#review-each-dataset-as-it-finishes)

The campaign review archive under `reports/datasets/<run-id>/details/` contains the individual receipt metadata, request measurements, document measurements, and sampled timelines. Publish one dataset at a time to get a manifest with its file hashes and coverage across all four cohorts. [Publish the detailed campaign measurements](../docs/vllm-native-pd-runbook.md#publish-the-detailed-campaign-measurements)
