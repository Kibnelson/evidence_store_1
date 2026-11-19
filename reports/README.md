# Reports and run artifacts

Start with the [earlier PD run review](prior-pd-review.md) for the existing experiment evidence and the instrumentation gaps it identifies.

Each new run gets a directory at `reports/runs/<run-id>/` with the source and topology manifest, replay receipt, raw JSONL events, profiler and hardware-counter artifacts, and summary. Preserve the raw event and output-quality receipts when publishing a summary so each reported time and byte total can be checked. Version the summaries and small receipts used to support comparisons.
