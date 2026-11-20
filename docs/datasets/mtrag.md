# MT-RAG request and chunk timeline

On the same 40 CLAPNQ turns, `previous-turn-first` restored **108 of 108 repeated evidence occurrences**, compared with **74** for the default prompt and **6** for the OpenClaw-style prompt. Ordinary prefill work on repeated evidence fell from **8,726 scheduled tokens** to **0** against the default. The Blend worker did more layerwise attention and MLP work and restored more KV from the host. The full NIXL payload remained **10,430,447,616 bytes** in both docs-first runs. [Default archive][default] · [OpenClaw archive][openclaw] · [Previous-turn-first archive][previous]

## Workload and repeat coverage

The MT-RAG human benchmark has four domains. We selected six complete CLAPNQ conversations of 7, 5, 9, 7, 6, and 6 turns. CLAPNQ had the highest repeat count among the eligible 40-turn subsets searched within each domain. Each selected turn runs in order, uses the saved reference conversation and retrieval rewrite, and retrieves six chunks from the full **183,408-row CLAPNQ Faiss HNSW index**. Each policy processes the same turns and original retrieval ranks. [Selection manifest](../../workloads/mtrag-multi-turn-repeat.json) · [Repeat workload selection](repeat-workloads.md)

| Domain | Corpus rows | Conversations | Turns | Highest repeats in a 40-turn subset |
| --- | ---: | ---: | ---: | ---: |
| CLAPNQ | 183,408 | 29 | 224 | 108 |
| Cloud | 72,442 | 26 | 205 | 90 |
| FiQA | 61,022 | 27 | 199 | 66 |
| Govt | 49,607 | 28 | 214 | 94 |

The 40 CLAPNQ turns contain **240 evidence occurrences** and **132 distinct versioned chunks**, for **108 repeats (45%)**. A repeat has the same `doc_id`, `chunk_id`, and `chunk_version` as an earlier occurrence. Of the 108 repeats, 93 had appeared within the same conversation and 27 had appeared in an earlier conversation. Twelve met both conditions. The source of a completed restore identifies the earlier store that supplied the KV, so each verified restore has one source scope. [Selection manifest](../../workloads/mtrag-multi-turn-repeat.json) · [Restore verification](../vllm-legacy-full-blend-tracing.md#how-we-verify-a-repeated-chunk)

The six chunks per turn came from **31 distinct source documents** over the run. In **32 of the 40 turns**, all six chunks came from a single document. This workload captures reuse of individual passage chunks within a document as the conversation advances. [Published chunk rows][default-chunks]

## Three prompt policies on one Blend worker

All three Slurm job 30511 runs used one Blend worker (`p0`) with its in-process LMCache host cache, three decode workers (`d0`–`d2`), Qwen3-4B-Instruct-2507, the same Faiss index and request order, and source tracing enabled. Each completed 40 dataset turns plus two controlled probes. Both paired comparisons against the default verified matching ranked evidence. NIXL READ counts and bytes reconciled against native counters on each worker. The tables below use the 40 dataset turns. [Default archive][default] · [OpenClaw archive][openclaw] · [Previous-turn-first archive][previous]

| Policy | Prompt layout and evidence order |
| --- | --- |
| `default` | Instructions, evidence in Faiss order, saved reference history and current question, answer cue. |
| `openclaw` | Instructions, saved reference history and current question, evidence in Faiss order, answer cue. |
| `previous-turn-first` | Keeps the default layout. Within the current retrieval selection, puts chunks from the previous successful turn of the same conversation first, in that turn's prompt order. Other earlier chunks follow in first-seen order, then new chunks follow in Faiss order. |

The legacy connector's layerwise lookup ends at the first missing segment. Moving history and query before evidence can put a changing segment ahead of cached chunks. Putting previously seen chunks before unseen chunks can bring those restores within reach. The OpenClaw-style adaptation here uses CLAPNQ reference history and this legacy connector. [Prompt policy definitions](../vllm-legacy-full-blend-runbook.md#compare-prompt-policies) · [Legacy lookup behavior](../slide-4-pd-measurement-plan.md#what-the-blend-settings-enable)

## Compute and movement across the timeline

The client joins each occurrence from the Faiss row to the final `[start_token, end_token)` interval. The report joins that interval with the gateway route, vLLM request IDs, scheduled work, LMCache receipts, and the NIXL READ to the decoder. Forming a repeat from evidence identity and proving an actual restore are separate operations. [Trace joins and measurement scope](../vllm-legacy-full-blend-tracing.md)

| Stage | What we measure for new and repeated evidence |
| --- | --- |
| Embed and search | Query embedding and Faiss HNSW elapsed time are request-wide. Each of the 40 turns performs live retrieval in every policy. |
| Read and build prompt | The client records each chunk's corpus row read, append, and offset mapping. All three runs read all 240 rows, totaling 134,249 serialized row bytes on new occurrences and 117,855 on repeats. The tokenizer processes the assembled prompt on every request. |
| Route through gateway | Receive, validate, dispatch to P, receive the KV handoff, then dispatch to D. These times apply to the request. |
| Compute on P | Intersect ordinary vLLM scheduled token ranges with each chunk span. Separately count positions reaching Blend attention and MLP across layers. CUDA events time the full ordinary forward batch. |
| Move KV between P GPU and host | Completed D2H stores and H2D restores carry bytes, token ranges, and timestamps. A verified repeat restore links the H2D receipt to an earlier same-worker store and a matching token slice. |
| Move KV from P to D | D uses NIXL to read the full assembled prompt KV with decode prefix caching off. Whole blocks inside a chunk span contribute to that chunk's exclusive byte count. The full payload also includes shared, boundary, and outside-evidence blocks and padding. |
| Decode and stream | D scheduled 0 ordinary tokens inside evidence spans in all three cases. Each run generated 1,280 output tokens across the 40 turns. CUDA events time the decode forward batch. Scheduled ranges and the first and last streamed token timestamps are per request. |

| Evidence-path measurement | Evidence occurrences | `default` | `openclaw` | `previous-turn-first` |
| --- | --- | ---: | ---: | ---: |
| Count | New / repeat | 132 / 108 | 132 / 108 | 132 / 108 |
| Prompt token positions before attributing work | New / repeat | 28,760 / 25,001 | 28,760 / 25,001 | 28,760 / 25,001 |
| Verified LMCache restores | Repeat | 74 | 6 | 108 |
| Within / across conversation source | Restored repeats | 57 / 17 | 0 / 6 | 81 / 27 |
| Restores to a shifted prompt offset | Repeat | 54 | 0 | 55 |
| Ordinary P scheduled tokens | New | 28,760 | 28,760 | 28,760 |
| Ordinary P scheduled tokens | Repeat | 8,726 | 23,389 | 0 |
| Blend attention and MLP token-layer visits | Repeat | 100,205 | 9,977 | 153,976 |
| LMCache D2H bytes | New | 4,240,834,560 | 4,240,834,560 | 4,240,834,560 |
| LMCache H2D bytes | Repeat | 2,399,846,400 | 237,699,072 | 3,686,547,456 |
| Exclusive NIXL bytes | New | 3,954,180,096 | 3,947,102,208 | 3,949,461,504 |
| Exclusive NIXL bytes | Repeat | 3,451,650,048 | 3,449,290,752 | 3,456,368,640 |

All three runs recorded zero evidence H2D and Blend attention/MLP visits on first appearances, and zero evidence D2H bytes on repeats. Ordinary tokens and token-layer visits measure work at different scopes. A cached row can still undergo Blend repair. The check layer can also project and score more positions than the attention and MLP visits count. [Default chunk rows][default-chunks] · [OpenClaw chunk rows][openclaw-chunks] · [Previous-turn-first chunk rows][previous-chunks]

All six OpenClaw-style restores came from one exact prompt repeat across conversations. The first turns of `0208bf26ec357a803445290fa88a2e9e` and `cfbcfa3e382f12aed450b10f87a72aeb` had the same 1,647-token prompt hash and identical evidence offsets. The second of those requests restored all six chunks at their original offsets. [OpenClaw request rows][openclaw-requests] · [OpenClaw chunk rows][openclaw-chunks]

| Whole-request measurement over 40 turns | `default` | `openclaw` | `previous-turn-first` |
| --- | ---: | ---: | ---: |
| LMCache GPU → host payload bytes | 6,598,361,088 | 6,598,361,088 | 6,598,361,088 |
| LMCache host → GPU payload bytes | 2,459,860,992 | 286,507,008 | 3,746,562,048 |
| NIXL P → D payload bytes | 10,430,447,616 | 10,439,884,800 | 10,430,447,616 |
| Prefill HTTP p50 / p95 (ms) | 344.2 / 548.1 | 391.6 / 544.5 | 309.2 / 457.7 |
| Client TTFT p50 / p95 (ms) | 1,827.0 / 2,439.1 | 1,851.5 / 2,374.9 | 1,785.7 / 2,427.5 |

Compared with the default, `previous-turn-first` added 1,286,701,056 bytes of whole-request H2D and eliminated the 8,726 ordinary P token positions on repeat evidence. D still read the full prompt KV. Reordering the evidence changed the blocks attributed exclusively to repeat spans within that payload. Client TTFT includes retrieval and prompt setup. The paired median TTFT change was −22.6 ms, with a p95 change of +60.9 ms across the 40 matched turns. These are trace-on timings from one run per policy. [Default request rows][default-requests] · [OpenClaw request rows][openclaw-requests] · [Previous-turn-first request rows][previous-requests]

### Measured stage timings

Times are nearest-rank medians in milliseconds for the 40 natural turns per policy. Where movement occurred on a subset of turns, the median covers that subset and the table gives its size. The restore counts include any LMCache segment restored, such as the instruction header. [Default request rows][default-requests] · [OpenClaw request rows][openclaw-requests] · [Previous-turn-first request rows][previous-requests]

| Stage | Type | `default` (p50 ms) | `openclaw` (p50 ms) | `previous-turn-first` (p50 ms) | Total bytes (policy order) |
| --- | --- | ---: | ---: | ---: | --- |
| Embed query | Compute | 839.049 | 842.598 | 839.561 | — |
| Search Faiss HNSW | Compute | 4.980 | 5.054 | 5.122 | — |
| Materialize evidence | Compute + movement | 0.177 | 0.171 | 0.180 | 252,104 / 252,104 / 252,104 |
| Render and map prompt | Compute | 4.297 | 4.114 | 4.230 | — |
| Tokenize prompt | Compute | 3.520 | 3.149 | 3.240 | — |
| Validate at gateway | Compute | 0.286 | 0.293 | 0.289 | — |
| Select P and build HTTP request | Routing + compute | 0.272 | 0.266 | 0.260 | — |
| Send prompt token IDs to P and receive response | Compute + movement | 344.229 | 391.574 | 309.221 | Raw trace only |
| LMCache restore on P (host → GPU) | Movement | 156.634 (39/40) | 155.799 (39/40) | 154.890 (39/40) | 2,459,860,992 / 286,507,008 / 3,746,562,048 |
| Restore start to Blend completion | Compute + movement | 157.340 (39/40) | 156.495 (39/40) | 155.609 (39/40) | — |
| Ordinary P CUDA forward | Compute | 183.256 | 217.180 | 143.769 | — |
| LMCache store on P (GPU → host) | Movement | 181.861 (39/40) | 221.309 (39/40) | 143.193 (39/40) | 6,598,361,088 / 6,598,361,088 / 6,598,361,088 |
| Route from P response to D dispatch | Routing + compute | 0.535 | 0.519 | 0.521 | — |
| NIXL READ (P GPU → D GPU) | Movement | 544.560 | 564.376 | 580.409 | 10,430,447,616 / 10,439,884,800 / 10,430,447,616 |
| All D CUDA forwards | Compute | 1,281.556 | 1,288.936 | 1,280.152 | — |
| D dispatch to client first token | Latency | 605.368 | 627.278 | 665.371 | — |
| Stream from first to last token | Compute + movement | 1,329.219 | 1,337.766 | 1,334.089 | Raw trace only |
| Client time to first token | Latency | 1,827.017 | 1,851.451 | 1,785.699 | — |

Byte totals sum the 40 requests per policy in the same left-to-right order as the timing columns. Materialization counts serialized corpus rows, and LMCache and NIXL count completed KV payloads. A dash marks a stage without a separate payload count. The raw gateway trace records HTTP body and stream bytes, which the compact archive omits. [Gateway byte counters](../../src/evidencestore/gateway.py#L108)

After validation, the gateway selects P and builds a JSON request with all prompt token IDs, including repeated evidence. The median prompt was **1,646 tokens** for `default` and `previous-turn-first` and **1,647 tokens** for `openclaw`. The send row measures the full HTTP round trip, from gateway dispatch through receipt and validation of P's handoff response. Raw gateway traces also record serialized request-body bytes per request. [Gateway dispatch and response events](../../src/evidencestore/gateway.py#L95)

[Trace review TODO for gateway → P delivery time and bytes](../vllm-legacy-full-blend-tracing.md#follow-ups-from-the-end-to-end-trace-review)

Read each row independently. Prompt rendering includes tokenization, the P round trip covers the P stages below it, and client time to first token starts before retrieval. LMCache timings are host intervals through the completion fence. The restore-to-Blend row includes the restore interval, and the chunk table above counts Blend attention/MLP work. The NIXL row spans host submission to completion. The CUDA rows use forward batch events, with all 32 D forwards summed per request. These were trace-on runs with GPU synchronization. [Trace timing and attribution](../vllm-legacy-full-blend-tracing.md#compute-and-timeline-stages)

### Estimated savings from sending changed KV rows

The NIXL row above measures the full P → D payload. This sensitivity table treats verified repeat-evidence KV restores on P as candidates. It assumes D already holds a compatible base for every candidate, including cross-conversation candidates, and that replacement rows can be packed with zero metadata overhead. Each scenario shows **payload reduction / ideal bandwidth-bound speedup** across 40 turns. The measured comparison would start with the previous-turn base on the same D and classify the final post-Blend rows against it. [Final-row method](../vllm-legacy-full-blend-tracing.md#calculate-actual-bytes-from-the-final-blend-output) · [Diff ratio and speedup](../vllm-legacy-full-blend-tracing.md#predict-the-speedup-and-break-even-point)

| Policy | Measured full bytes | Candidate bytes | 10% of candidate KV sent | 50% sent | 90% sent |
| --- | ---: | ---: | ---: | ---: | ---: |
| `default` | 10,430,447,616 | 2,399,846,400 | 20.71% / 1.26× | 11.50% / 1.13× | 2.30% / 1.02× |
| `openclaw` | 10,439,884,800 | 237,699,072 | 2.05% / 1.02× | 1.14% / 1.01× | 0.23% / 1.00× |
| `previous-turn-first` | 10,430,447,616 | 3,686,547,456 | 31.81% / 1.47× | 17.67% / 1.21× | 3.53% / 1.04× |

For `previous-turn-first` at 10% changed, the model sends about **7.11 GB of 10.43 GB**. At 100% changed, every policy converges to its full-payload measurement. To approximate the earlier **previous-turn-only** base policy, restrict candidates to those whose chunk also appeared in the preceding turn of the same conversation. With sticky D placement and that base retained, the 10%-changed case avoids **13.00%** for `default`, **0%** for `openclaw`, and **17.58%** for `previous-turn-first`. The identity join estimates which rows could have a base. A final KV comparison would determine how many actually match. [Default chunk rows][default-chunks] · [OpenClaw chunk rows][openclaw-chunks] · [Previous-turn-first chunk rows][previous-chunks]

## A repeat the default computed again

Turn 7 of conversation `0208bf26ec357a803445290fa88a2e9e` (`pd-edc8b10503666f40e94b1c82`) retrieved six chunks. In the default prompt, ranks 1 and 2 restored, rank 3 was new, and repeated ranks 4 and 6 ran as ordinary prefill. This ordering matches the connector's first-miss behavior. Under `previous-turn-first`, prompt order became `1, 4, 6, 2, 3, 5` and all four repeats restored. Rank 4, `mtrag:clapnq:chunk:652868320e935c479500505cb2d7e49084e930eff482a6265d3139916b7c9d56`, restored from the same worker's turn 6 store at `[1309, 1538)`. The rank and the 1,131-byte corpus row identify the same 229-token chunk in all three runs. [Default chunk rows][default-chunks] · [OpenClaw chunk rows][openclaw-chunks] · [Previous-turn-first chunk rows][previous-chunks]

| Rank 4 chunk measurement | `default` | `openclaw` | `previous-turn-first` |
| --- | ---: | ---: | ---: |
| Final token span | `[714, 943)` | `[1154, 1383)` | `[273, 502)` |
| Corpus row read / append / offset mapping (µs) | 3.140 / 0.710 / 58.930 | 2.960 / 0.550 / 61.661 | 3.350 / 1.030 / 57.161 |
| Ordinary P scheduled tokens | 229 | 229 | 0 |
| Blend attention and MLP token-layer visits | 0 | 0 | 684 |
| LMCache host → GPU bytes | 0 | 0 | 33,767,424 |
| LMCache GPU → host bytes | 0 | 0 | 0 |
| NIXL bytes exclusive to this chunk | 30,670,848 | 30,670,848 | 30,670,848 |

The 684 Blend visits comprise all 229 positions at layer 0 and 13 positions at each of the next 35 layers. The first-seen rank 3 chunk still scheduled 208 ordinary P positions and wrote 30,670,848 KV bytes to the host in all three prompts. The decoder `d0` read **273,678,336 bytes** for the whole request in 116 blocks, in every policy. [Default request rows][default-requests] · [OpenClaw request rows][openclaw-requests] · [Previous-turn-first request rows][previous-requests]

| Turn 7 milestone (ms from client start) | `default` | `openclaw` | `previous-turn-first` |
| --- | ---: | ---: | ---: |
| HNSW search | 845.375–852.306 | 851.295–858.347 | 845.428–852.372 |
| Prompt render | 852.572–856.685 | 858.621–862.581 | 852.689–856.690 |
| Gateway dispatch to P | 858.010 | 863.920 | 858.029 |
| Request H2D complete / Blend complete | 1,012.461 / 1,013.631 | 1,023.824 / 1,024.537 | 1,027.025 / 1,028.637 |
| Ordinary P forward, host start and end | 1,013.991–1,189.036 | 1,024.715–1,253.299 | 1,029.369–1,153.455 |
| Ordinary P forward CUDA duration | 183.256 | 239.497 | 129.662 |
| P handoff ready / Gateway dispatch to D | 1,200.668 / 1,206.546 | 1,267.524 / 1,271.218 | 1,162.485 / 1,166.445 |
| D NIXL READ submission and completion | 1,218.348–1,839.771 | 1,279.728–1,906.378 | 1,175.178–1,781.718 |
| Client first / last token | 1,891.415 / 3,216.555 | 1,958.565 / 3,302.348 | 1,833.634 / 3,165.498 |

The request-wide OpenClaw H2D milestone covers a restored 8-token instruction segment. The rank 4 row records zero H2D. The chunk spans and completed receipts identify exactly which KV moved within a request. [Default request rows][default-requests] · [OpenClaw request rows][openclaw-requests] · [Previous-turn-first request rows][previous-requests]

[default]: ../../reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30511-20261120T142442Z/1p3d/blend/legacy-blend-summary.json
[openclaw]: ../../reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30511-20261120T142806Z/1p3d/blend/legacy-blend-summary.json
[previous]: ../../reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30511-20261120T150130Z/1p3d/blend/legacy-blend-summary.json
[default-chunks]: ../../reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30511-20261120T142442Z/1p3d/blend/legacy-chunks.jsonl.gz
[openclaw-chunks]: ../../reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30511-20261120T142806Z/1p3d/blend/legacy-chunks.jsonl.gz
[previous-chunks]: ../../reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30511-20261120T150130Z/1p3d/blend/legacy-chunks.jsonl.gz
[default-requests]: ../../reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30511-20261120T142442Z/1p3d/blend/legacy-requests.jsonl.gz
[openclaw-requests]: ../../reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30511-20261120T142806Z/1p3d/blend/legacy-requests.jsonl.gz
[previous-requests]: ../../reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30511-20261120T150130Z/1p3d/blend/legacy-requests.jsonl.gz
