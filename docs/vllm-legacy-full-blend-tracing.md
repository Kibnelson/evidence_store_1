# Full Blend tracing for compute and KV movement

This reference describes the vLLM `0.23.0` and in-process LMCache `0.5.0` runs. The Blend worker B is the prefill worker, labeled P in the traces. Each B uses its own CPU KV cache. The decode worker D pulls the assembled prompt KV over native NIXL with decode prefix caching disabled. The `default`, `seen-first`, and `openclaw` prompt policies use the same measurement hooks. [Runbook and source pins](vllm-legacy-full-blend-runbook.md) · [PD instrumentation plan](pd-instrumentation.md)

## How the trace follows one chunk

The runtime builder restores the detached vLLM and LMCache sources and applies the tracing overlay to the scratch build. The overlay adds events at the input processor, scheduler, GPU forward, LMCache layerwise store and restore, Blend layer execution, and native NIXL submission and completion. The launcher maps `trace=on` to `EVIDENCESTORE_PD_TRACE=1` on the workers and `EVIDENCESTORE_TRACE_REQUESTS=1` for the client and gateway. It also passes `--trace-chunk-stages` to the live replay. Trace-off runs keep the response receipts and aggregate NIXL counters. [Runtime builder](../scripts/vllm_pd_legacy_blend_runtime.sh#L102) · [Source hooks](../scripts/vllm_pd_legacy_trace_overlay.py#L37) · [Trace flags](../scripts/vllm_pd_cluster.py#L246) · [Live replay launcher](../scripts/vllm_pd_legacy_blend.py#L55)

The client associates a stable `request_key` and `request_id` with each question or turn. For each gateway attempt, the gateway records which P and D handled it and the header IDs used to dispatch to them. vLLM records the mapping from each header ID to its native request ID. The report uses that mapping to join client, gateway, vLLM, LMCache, and NIXL events. The event timestamps are monotonic nanoseconds on the same Slurm node. Batch and transfer IDs associate events with a forward or a NIXL READ. [Client identifiers](../scripts/vllm_pd_replay.py#L191) · [Gateway routing](../src/evidencestore/gateway.py#L96) · [Trace join](../src/evidencestore/pd_report.py#L190)

Faiss returns row labels and scores. The client resolves each label to a corpus record with `doc_id`, `chunk_id`, `chunk_version`, and the retrieval rank. Prompt rendering maps each retained occurrence to a half-open token interval `[start_token, end_token)` in the exact token sequence sent to vLLM. If a policy moves a chunk, its retrieval rank and identity stay attached to its new interval. The report intersects work and transfer ranges with those intervals. [Corpus identity](../src/evidencestore/retrieval.py#L52) · [Corpus materialization](../src/evidencestore/retrieval.py#L396) · [Final token spans](../src/evidencestore/retrieval.py#L525) · [Chunk attribution](../scripts/vllm_pd_legacy_blend_report.py#L158)

## Compute and timeline stages

| Stage | What the trace measures | Scope |
| --- | --- | --- |
| Query embedding and Faiss HNSW search | Start and end for embedding, search, and number of returned rows. | One query. The search interval is shared by its returned chunks. [Live replay](../scripts/vllm_pd_replay.py#L220) |
| Corpus materialization | The interval for each row seek and read, its Faiss row label and byte offset, and the serialized row size. The outer materialization interval covers the broader read, parse, and ranking stage. | One row for `corpus_row_read`. One request for the outer interval. [Corpus reader](../src/evidencestore/retrieval.py#L396) |
| Prompt assembly and tokenization | Time each chunk takes to append, then time to map its character range to final token offsets. Whole-prompt timings separately cover wrapper assembly, joining, and tokenization. Retain the accepted attempt and the chunks dropped to fit the budget. | Append and offset mapping are per chunk. Tokenization runs on the whole prompt for each render attempt. [Prompt renderer](../src/evidencestore/retrieval.py#L447) |
| Gateway and P dispatch | Gateway receive, prompt validation, worker choice, P dispatch, and receipt of the P response with native KV coordinates. | Host milestones for the request. P HTTP time includes work and waiting inside P. [Gateway](../src/evidencestore/gateway.py#L76) |
| Ordinary model work on P | `engine_token_work` logs each scheduled token range before vLLM advances its computed-token count. The report intersects the range with each evidence span and records `scheduled_tokens_by_role.P`. | Scheduled token positions for the chunk. CUDA events around the ordinary forward time the entire model batch and carry its request membership. [Scheduler and forward hooks](../scripts/vllm_pd_legacy_trace_overlay.py#L48) · [CUDA timer](../scripts/vllm_pd_legacy_trace_events.py#L81) |
| Blend on P | The layerwise hook records positions after each layer's attention and MLP. The report intersects them with the evidence spans and writes `blend_computed_tokens_by_call_and_layer`. | Token positions per layer and Blend call. Earlier layers can visit more positions than later repair layers. The check layer's projection and scoring can also visit more positions than the attention and MLP count. [Blend hook](../scripts/vllm_pd_legacy_trace_overlay.py#L159) · [Layer range receipt](../scripts/vllm_pd_legacy_trace_events.py#L111) |
| NIXL and decode | P handoff ready, D dispatch, READ submission and completion, D scheduled token ranges and CUDA forwards, then first and last tokens at the client. | D pulls prompt KV before generation. The per-chunk D work count covers positions inside the chunk's prompt span. The request timeline also includes generation. [NIXL and handoff hooks](../scripts/vllm_pd_legacy_trace_overlay.py#L70) · [Client token timestamps](../scripts/vllm_pd_replay.py#L165) |

Read ordinary scheduled tokens and Blend token-layer positions separately. The first counts tokens handed to an ordinary forward. The second counts visits to attention and MLP across layers. Whole-batch CUDA duration and host intervals use different clocks and measure different scopes. The trace includes CUDA synchronization for diagnostic timing. [Measurement scope in the report](../scripts/vllm_pd_legacy_blend_report.py#L441)

The paired report gives p50 and p95 for prompt rendering, client TTFT, dispatch-to-first-token time, and P HTTP time. Client TTFT starts before live retrieval. Dispatch-to-first-token starts after prompt assembly. The paired differences match requests by key and preserve their order. [Paired timing comparison](../scripts/vllm_pd_legacy_blend_report.py#L505)

## What movement we measure

| Path | Bytes and timing recorded | How we attribute it |
| --- | --- | --- |
| P GPU → P host CPU cache | `legacy_lmcache_d2h_complete` records completed KV stores. Bytes sum the LMCache memory-object sizes for the range across layers. The host interval runs through completion of the layerwise copy and storage-put path. | Record the saved token range, worker, request, and layer count. [Store hook](../scripts/vllm_pd_legacy_trace_overlay.py#L104) |
| P host CPU cache → P GPU | `legacy_lmcache_h2d_complete` records completed restores. Bytes sum the loaded memory-object sizes across layers. The interval ends at the native completion fence. | Record the old cached offset and the destination token range so the report can link the restore to a prior store. [Restore hook](../scripts/vllm_pd_legacy_trace_overlay.py#L123) |
| P GPU → D GPU via NIXL | The D worker records the READ ID and block geometry at submission. On completion it records `telemetry.totalBytes`, NIXL transfer and post durations, and the completion timestamp. | Verify the P request, block list, and full prompt coverage, including block padding. Reconcile successful READ bytes and count against the before/after `vllm:nixl_bytes_transferred` histogram sum and count per worker. [READ receipts](../scripts/vllm_pd_legacy_trace_events.py#L53) · [Handoff checks](../scripts/vllm_pd_legacy_blend_report.py#L88) · [Counter reconciliation](../src/evidencestore/pd_report.py#L683) |

NIXL assigns the bytes of a whole block to a chunk only when the block lies entirely inside that occurrence. Blocks that overlap evidence and another span are recorded as shared or boundary bytes. Blocks that miss all evidence spans are outside-evidence bytes. The report also retains an unmapped bucket for incomplete geometry. The full request payload includes these buckets and the chunk-exclusive bytes. LMCache and NIXL figures are completed KV payloads on separate hops. Filesystem and HTTP fields in the raw trace are serialized data sizes, and physical PCIe and network-interface counters are outside this report's measurement scope. [Block attribution](../src/evidencestore/pd_report.py#L260) · [Raw HTTP milestones](../src/evidencestore/gateway.py#L112)

## How we verify a repeated chunk

- **Repeated occurrence.** The same `(doc_id, chunk_id, chunk_version)` appeared in an earlier request in the cohort. The report also records whether it appeared on the same P. In MT-RAG it records repeats within and across conversations. [Repeat tracking](../scripts/vllm_pd_legacy_blend_report.py#L243) · [Conversation scopes](../scripts/vllm_pd_legacy_blend_report.py#L278)
- **Verified restore.** A completed H2D receipt identifies the current range and the old offset. The report checks for a prior same-worker D2H receipt at the old range and verifies that the restored tokens match that earlier token slice. A shifted restore additionally lands at a different prompt offset. [Restore lineage](../scripts/vllm_pd_legacy_blend_report.py#L138)
- **Repeat with ordinary work.** Keep the repeated occurrence in the report even when it has no restore. Its ordinary P token count still shows what vLLM scheduled. The legacy layerwise lookup can stop at an earlier miss. The current receipt measures completed movement and compute, so it can show that a repeat was computed again even when it cannot identify the lookup's exact miss or eviction reason. [Legacy lookup behavior](slide-4-pd-measurement-plan.md#what-the-blend-settings-enable)

## Example from MuSiQue job 30507

The same 126-token chunk `musique:chunk:2cc56cd5895a…` appeared in requests `pd-b03d30040e0043f8335d2fd0` and `pd-c2f03a9708aef30ef84e0cb5`. Both used `p0`, so the later restore can be linked to the earlier store. [Published request and chunk rows](../reports/smoke/musique-single-turn-40-legacy-full-blend-30507-20261120T033615Z/1p3d/blend/)

| Measurement for this chunk | First occurrence | Repeated occurrence |
| --- | ---: | ---: |
| Retrieval rank and prompt span | 2, `[150, 276)` | 3, `[368, 494)` |
| Ordinary P scheduled tokens | 126 | 0 |
| Blend attention and MLP positions | 0 | 126 at layer 0, then 9 per layer at layers 1–35 |
| LMCache GPU → host bytes | 18,579,456 | 0 |
| LMCache host → GPU bytes | 0 | 18,579,456 |
| NIXL bytes exclusive to the chunk | 16,515,072 | 16,515,072 |

In the repeated request, the chunk's 682-byte corpus row took 4.300 µs to read. Appending its text took 0.470 µs and mapping it to final token offsets took 36.951 µs. The enclosing prompt-render interval also includes shared work such as tokenizing the joined prompt. [Published chunk rows](../reports/smoke/musique-single-turn-40-legacy-full-blend-30507-20261120T033615Z/1p3d/blend/legacy-chunks.jsonl.gz)

| Repeated request milestone | Elapsed from client start |
| --- | ---: |
| Faiss HNSW search | 843.558–848.871 ms |
| Prompt render | 849.156–851.780 ms |
| Gateway dispatches to P | 853.051 ms |
| P completes the LMCache restore, then Blend | 1,043.034 ms and 1,044.340 ms |
| P ordinary forward host start and end | 1,044.726 ms and 1,142.071 ms |
| P advertises the assembled KV | 1,149.784 ms |
| D submits and completes the NIXL READ | 1,430.586 ms and 1,846.889 ms |
| Client receives the first token | 1,963.837 ms |

The repeated request transferred **176,947,200 bytes** over NIXL for the full 1,190-token prompt in 75 blocks of 16 slots. This includes 155,713,536 bytes exclusive to the six evidence chunks, 16,515,072 shared or boundary bytes, and 4,718,592 bytes outside evidence. The two requests have different prompt lengths. The chunk rows establish the change in work and cache movement for that occurrence, while the paired workload runs control which requests and evidence are compared. [Published request rows](../reports/smoke/musique-single-turn-40-legacy-full-blend-30507-20261120T033615Z/1p3d/blend/legacy-requests.jsonl.gz)

## Where to check a run

- `legacy-requests.jsonl` has the request timeline, full NIXL bytes and block coverage, and completed LMCache movements. `legacy-chunks.jsonl` has each evidence occurrence's preparation stages, scheduled positions, Blend work, restore lineage, and exclusive NIXL bytes. Published review packets compress these files as `.jsonl.gz`. [Archive format](vllm-legacy-full-blend-runbook.md#review-the-archive)
- `legacy-blend-summary.json` verifies the response and NIXL counts. In trace-on runs it also verifies one cold and one reordered probe per B, including a shifted restore on each B. These controls validate the tracing path even when a natural prompt policy has no document restores. [Report checks](../scripts/vllm_pd_legacy_blend_report.py#L374)
- `--compare DEFAULT_COHORT GATED_COHORT` checks matching workload, index, model, topology, request order, and ranked evidence. Read `repeated_evidence` for per-policy work and bytes on previously seen chunks, and `natural_requests` for whole-request transfer totals. Read `paired_gated_minus_default_latency_ms` for the timing differences. [Paired test commands](vllm-legacy-full-blend-runbook.md#run-both-pairs-in-one-allocation)
