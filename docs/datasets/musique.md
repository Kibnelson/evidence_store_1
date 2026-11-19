# MuSiQue request and chunk timeline

The full replay contains 2,417 MuSiQue requests and six retrieved chunks per request. The campaign runs the same prompts in each of four cohorts, pairing native NIXL with shared LMCache and local prefix caching in the 1P/3D and 2P/2D layouts. This review counts the benchmark requests once and compares each native request with its matched shared request. [MuSiQue results at `e1c8b50`](https://github.com/Kibnelson/evidence_store_1/tree/e1c8b50/reports/datasets/native-vllm-30453-20261119T031129Z/musique)

## Chunks and source documents per request

Every prompt contains six chunks from one to six distinct documents. We count source documents by deduplicating `doc_id` within each request. The four cohorts have identical counts for every prompt. [Request and chunk records](https://github.com/Kibnelson/evidence_store_1/tree/e1c8b50/reports/datasets/native-vllm-30453-20261119T031129Z/details/musique)

| Distinct documents in the prompt | Requests |
|---|---:|
| 6 | 1,719 |
| 5 | 388 |
| 4 | 163 |
| 3 | 100 |
| 2 | 33 |
| 1 | 14 |

Request `pd-3f6fc21c185a5da9f18974f6` has six chunks from five documents. The timeline example below has six chunks from six documents.

## Repeated chunks

| Measure | Count |
|---|---:|
| Retrieved chunk occurrences per cohort | 14,502 |
| Distinct chunk identities | 6,254 |
| Chunk identities found in two or more requests | 2,784 |
| All occurrences of those recurring identities, including their first uses | 11,032 |
| Repeat uses after the first | 8,248 (56.9% of all occurrences) |
| Requests containing a recurring chunk | 2,386 of 2,417 (98.7%) |
| Recurring identities observed at multiple prompt offsets | 2,747 of 2,784 (98.7%) |

An identity is the pair of `chunk_id` and `chunk_version`. The recurring-chunk view includes every occurrence of those identities. Frequency and prompt offset describe the workload. The request trace records whether P scheduled the chunk's tokens and how many KV bytes D pulled. [Request and chunk records](https://github.com/Kibnelson/evidence_store_1/tree/e1c8b50/reports/datasets/native-vllm-30453-20261119T031129Z/details/musique)

## Measured compute and movement

Each value is the reduction from native to shared within the same layout. Exclusive evidence bytes count KV blocks that fall wholly inside one chunk.

| Measurement | 1P/3D reduction | 2P/2D reduction |
|---|---:|---:|
| P scheduled prompt tokens across all requests | 17,472 | 13,216 |
| P scheduled tokens inside retrieved chunks | 2,864 | 1,995 |
| Completed P→D NIXL payload | 1,361.25 MiB | 1,694.25 MiB |
| Exclusive evidence NIXL payload | 83.25 MiB | 148.50 MiB |
| Requests with a reduction in exclusive evidence NIXL payload | 1 | 2 |

Recurring chunks accounted for all of the measured reductions in P evidence tokens and exclusive evidence bytes. Chunk identities that appeared once contributed zero reduction in those two measures. Most of the complete request transfer reduction was outside the evidence spans. [Byte and token summaries](https://github.com/Kibnelson/evidence_store_1/tree/e1c8b50/reports/datasets/native-vllm-30453-20261119T031129Z/musique)

## Timeline stages and measurement scope

The first four stages ran live in `1p3d-native`. The other three cohorts replayed its frozen prompts and kept the same chunk identities, ranks, and token spans. All four measured the gateway, vLLM, and response stages. [Replay setup](../../scripts/vllm_pd_full_campaign.sh#L110) · [Trace span definitions](../../src/evidencestore/pd_report.py#L28)

| Stage | Request, batch, or cohort measurement | Measurement for a specific chunk |
|---|---|---|
| Query embedding | CPU elapsed time for the query tokenizer and E5 encoder | One query embedding is shared by the six results. |
| Faiss HNSW search | Elapsed time for the query search | Rank, score, and stable chunk identity. |
| Read the retrieved chunks | Elapsed time to fetch all six corpus rows. The live trace also records their combined row bytes | `doc_id`, `chunk_id`, and `chunk_version` from each row. |
| Build and tokenize the prompt | One elapsed span for assembly, whole-prompt tokenization, evidence mapping, and prompt hashing | Start and end token positions in the finished prompt. Assembly and tokenizer timings are shared across the prompt. |
| Gateway validation and dispatch | Parse and validation time, selected P/D workers, P HTTP round trip, and HTTP payload bytes | The report checks the chunk's token span against the evidence mapping recorded by the gateway. HTTP payload bytes are recorded per request. |
| Prefill on P | Scheduler wait per request and CUDA forward time per batch | Scheduled prompt tokens and observed local or remote prefix matches that overlap the chunk. |
| LMCache store or load | DMA staging bytes per cohort and direction. The request trace records the selected connector and remote KV wait where applicable | The selected connector identifies an LMCache match on the request. Its overlap with the chunk is measured in tokens. Staging byte totals are cohort-level. |
| NIXL handoff from P to D | D's submit-to-completion span, remote KV wait, and completed payload bytes. Worker counter deltas reconcile with the traces | Exact bytes for completed KV blocks lying wholly inside the chunk. Blocks crossing evidence boundaries are counted separately. |
| Decode on D | Scheduler wait per request, scheduled decode tokens, and CUDA forward time per batch | Scheduled prompt tokens that overlap the chunk, plus local and remote prefix matches within its span. D scheduled zero evidence tokens across MuSiQue. |
| Stream the response | First token, last token, completion, and HTTP stream bytes for the request | All evidence chunks share the response milestones. |

The join uses the request ID, chunk identity, and prompt token range. The three different units answer different questions. Elapsed time shows when a stage ran or waited, scheduled tokens show how much chunk work was assigned, and completed KV bytes show how much data moved. [Request measurements](https://github.com/Kibnelson/evidence_store_1/tree/e1c8b50/reports/datasets/native-vllm-30453-20261119T031129Z/details/musique) · [Token and byte attribution](../../src/evidencestore/pd_report.py#L220)

### LMCache movement in MuSiQue

These are dataset-window DMA staging totals across all worker devices. Each direction uses the staging counter once. The kernel counters cover a separate phase of the same transfer path. [MuSiQue summaries](https://github.com/Kibnelson/evidence_store_1/tree/e1c8b50/reports/datasets/native-vllm-30453-20261119T031129Z/musique) · [LMCache counter definition](https://github.com/LMCache/LMCache/blob/042d47abdef0d460cf2cb0d5720e8a59280cda8a/lmcache/v1/mp_observability/subscribers/metrics/phase_timing.py)

| Shared cohort | LMCache GPU → host | LMCache host → GPU | P selected LMCache | D selected LMCache |
|---|---:|---:|---:|---:|
| 1P/3D | 209,484 MiB | 0 MiB recorded | 0 | 0 |
| 2P/2D | 209,484 MiB | 216 MiB | 3 requests, 1,536 matched tokens | 0 |

In both shared cohorts, D selected NIXL for all 2,417 requests. The LMCache staging figures describe the host cache path. The NIXL figures describe the P→D handoff.

## One request through the timeline

Request `pd-753a6befacecbe7e8dffdfb8` contains 735 prompt tokens and six recurring chunks. Its prompt is identical across the four cohorts.

| Layout and cache mode | P prompt tokens scheduled | KV blocks D pulled | Completed NIXL payload | Dispatch to first token |
|---|---:|---:|---:|---:|
| 1P/3D native | 735 | 46 | 103.50 MiB | 425.70 ms |
| 1P/3D shared | 15 | 1 | 2.25 MiB | 891.56 ms |
| 2P/2D native | 735 | 46 | 103.50 MiB | 449.30 ms |
| 2P/2D shared | 223 | 1 | 2.25 MiB | 271.86 ms |

The sixth chunk, `a3607e1a8820…`, appears in six requests and occupies positions `[489, 733)` in this prompt. P scheduled 244 of its tokens in both native runs, 13 in 1P/3D shared, and 221 in 2P/2D shared. Bytes attributed exclusively to this chunk fell from 31.50 MiB to zero in both layouts. D scheduled one final prompt position and 31 decode positions in every arm. [Paired request archives](https://github.com/Kibnelson/evidence_store_1/tree/e1c8b50/reports/datasets/native-vllm-30453-20261119T031129Z/details/musique)

## Follow a chunk through the stages

For the sixth chunk in the example, Faiss recorded a score of 0.8144 and document `a611647dec96…`. The prompt span is `[489, 733)` in all four arms. The live request took 1.41 ms for Faiss, 0.55 ms to read the six chunks, and 3.37 ms to build and tokenize the prompt. These durations cover the request. [Paired request archives](https://github.com/Kibnelson/evidence_store_1/tree/e1c8b50/reports/datasets/native-vllm-30453-20261119T031129Z/details/musique)

| Stage | Native, both layouts | 1P/3D shared | 2P/2D shared |
|---|---:|---:|---:|
| P scheduled chunk tokens | 244 | 13 | 221 |
| P local prefix match in the chunk | 0 | 231 | 0 |
| P LMCache prefix match in the chunk | 0 | 0 | 23 |
| D local prefix match in the chunk | 0 | 231 | 231 |
| D NIXL prefix match in the chunk | 244 | 13 | 13 |
| NIXL bytes exclusive to the chunk | 31.50 MiB | 0 MiB | 0 MiB |
| D scheduled tokens in the chunk | 0 | 0 | 0 |

In 2P/2D shared, P selected LMCache for a 512-token prefix. The overlap with this chunk is `[489, 512)`, which accounts for its 23 matched tokens. P's remote KV wait was 6.50 ms for that request. The measured LMCache bytes appear in the cohort staging totals above.

In both shared layouts, D still pulled a 2.25 MiB NIXL block at `[720, 736)`. Its first 13 positions lie inside this chunk and the rest cross its end. The request accounts for the block as boundary traffic, so the exclusive count for this chunk is zero.

## Reading the timeline

Browse all request and chunk records in the four `requests.jsonl.gz` files. The comparison aligns each request at its client start and uses dispatch, first token, last token, and response completion as milestones. Internal stages such as the NIXL read and remote-KV wait have separate elapsed durations and can overlap. The eight `timeline.json` examples per cohort have timestamped internal events. A concurrent waterfall across the entire run requires the raw worker and gateway traces. [Detailed archives and samples](https://github.com/Kibnelson/evidence_store_1/tree/e1c8b50/reports/datasets/native-vllm-30453-20261119T031129Z/details/musique)

## Chunk-capture smoke before a rerun

Start with MuSiQue in 2P/2D. Each fresh cohort runs four serial controls and seven live requests. The controls cover a cold prompt, an exact repeat on the same workers, an exact repeat sent to the other prefiller, and a repeat with the same chunks reordered. The live sample includes MuSiQue requests that retrieved the same chunk at different token positions. The full retrieval index stays in use so the smoke exercises the same HNSW path. [Smoke driver](../../scripts/vllm_pd_chunk_smoke.sh)

The new trace records each retained chunk's corpus-row read and bytes, prompt append, and offset mapping. It joins on the `doc_id`, `chunk_id`, and `chunk_version` returned by corpus lookup. Faiss row numbers record which rows were read. The same collector accepts identifiers from all supported datasets. Prompt join and tokenization cover the complete prompt and preserve discarded attempts. Scheduled P/D operations retain their step IDs and token ranges. The transfer ledger records completed NIXL blocks and LMCache staging bytes, with the overlap of every mapped block and evidence occurrence. A block crossing evidence spans has one byte total and references every affected chunk. [Chunk report](../../src/evidencestore/pd_chunk_report.py)

### Run inside the Slurm allocation

Use the fresh-job runbook to allocate four GPUs, clone and build on node-local scratch, stage or reuse the MuSiQue index, and run the smoke. [MuSiQue chunk smoke from a fresh login shell](../vllm-native-pd-runbook.md#musique-chunk-smoke-from-a-fresh-login-shell)

### What must pass

- Every live evidence occurrence has matching read, append, and offset-mapping records, with the same prompt span used by the gateway and workers. The paired cohorts receive the same prompt tokens and evidence.
- The cold native probes cover the full prompt via NIXL. In the shared cohort, the exact warm request has less prefill work and less NIXL movement. The reordered request shows work and movement returning for recurring evidence.
- The cross-prefill request selects LMCache and completes a measured host-to-GPU restore with verified positions overlapping evidence. NIXL and LMCache completed byte totals reconcile with their respective counter snapshots.
- The live trace contains a recurring chunk at a changed prompt position. Its record references the preceding request and preserves the preparation work, scheduled operations, cache observations, and movement for the new occurrence.

Review `chunk-review/summary.json`, `requests.jsonl`, `chunks.jsonl`, and `kv-transfers.jsonl` in both printed cohort directories. The full rerun should start with MuSiQue after both smoke receipts pass. Keep each subsequent dataset in its own run directory so its measurements can be checked and published separately.
