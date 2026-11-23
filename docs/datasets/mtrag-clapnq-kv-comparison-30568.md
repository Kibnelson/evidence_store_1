# MT-RAG CLAPNQ KV storage comparison

Job 30568 captured six CLAPNQ conversations with 40 turns under 1B/3D Full Blend and `previous-turn-first`. Each conversation started with a fresh P cache. The capture contains 2,888 complete 16-token BF16 page stores and 1,220 observed loads. Each page spans all 36 layers and contains 2.25 MiB of K and V data. [Campaign index at `a7b976f`](https://github.com/Kibnelson/evidence_store_1/blob/a7b976f/reports/campaigns/clapnq-30568.json)

## What the six conversations show

**Independent compression produced the larger storage gain.** The archived per-page gzip frames use 18.70% fewer bytes than the raw pages. The traced exact-row patches save just 0.15% of raw bytes across all six captures, even when their bases remain available.

| Complete-page store representation | Bytes across 2,888 stores | Reduction from raw |
| --- | ---: | ---: |
| Raw KV | 6,813,646,848 | 0% |
| Raw KV with traced exact-row patches where smaller | 6,803,538,944 | 0.15% |
| Independently compressed page files in the capture | 5,539,675,173 | 18.70% |

The shadow tracer selected 556 smaller row patches against raw pages. Eighty of those page versions had later demand, yielding 110 byte-verified shadow reconstructions. The selected representation includes patch metadata. None of the 556 row patches beat the corresponding independent gzip frame. With a 5% minimum per-page saving, none qualified even against raw storage.

The 110 shadow reconstructions had a median apply time of 1.80 ms with the raw base available to the tracer. The modeled cold read of base plus patch totals **494.2 MiB**, compared with **247.5 MiB** for reading the same pages raw. This is approximately 2× the bytes for those 110 loads. [Example page report for the nine-turn capture](https://github.com/Kibnelson/evidence_store_1/blob/a7b976f/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30568-20261123T021826Z/1p3d/blend/legacy-kv-page-report.json)

### Cache hits at a fixed budget

The replay applies each conversation's chronological store and load events to a fresh logical cache, with LRU eviction and refill from the captured page on a miss. The two independent baselines use measured raw or gzip frame lengths plus a 32-byte version ID. The row-patch model adds one-hop dependencies using the selected bases and measured patch lengths from the trace. A patch must save at least 5% per frame. All three row-patch policies therefore had the raw baseline's hit counts. Adding that same row-patch option to the gzip baseline also left its hit counts equal to the independent gzip run.

| Cache budget per conversation | Raw, or raw with the traced row patches | Independent gzip | Additional gzip hits |
| ---: | ---: | ---: | ---: |
| 128 MiB | 85 / 1,220 | 386 / 1,220 | 301 |
| 256 MiB | 757 / 1,220 | 855 / 1,220 | 98 |
| 512 MiB | 982 / 1,220 | 1,072 / 1,220 | 90 |
| 768 MiB | 1,116 / 1,220 | 1,205 / 1,220 | 89 |
| 1,024 MiB | 1,220 / 1,220 | 1,220 / 1,220 | 0 |

At 1,024 MiB, where both baselines hit all 1,220 loads, modeled encoded read volume is 2.68 GiB raw and 2.18 GiB with gzip. These are host-side frame bytes. After reconstruction each complete page still contains its original 2.25 MiB of KV.

| Conversation | Turns | Complete loads | First tested budget for all hits, raw | First tested budget for all hits, gzip |
| --- | ---: | ---: | ---: | ---: |
| `0208bf26` | 7 | 183 | 864 MiB | 704 MiB |
| `1534a095` | 5 | 213 | 184 MiB | 148 MiB |
| `a2698b29` | 9 | 154 | 656 MiB | 536 MiB |
| `a3f07e9e` | 7 | 334 | 964 MiB | 784 MiB |
| `c1e53df9` | 6 | 115 | 180 MiB | 148 MiB |
| `cfbcfa3e` | 6 | 221 | 932 MiB | 756 MiB |

The first full-hit budgets use a coarse sweep plus 4-MiB steps in the preceding interval. Each row describes the smallest tested budget, so finer thresholds are possible. The budget measures committed complete-page frames. The capture's boundary fragments, reconstruction scratch, and GPU allocator are outside that budget.

### What moved in the live run

Existing Blend restored all 93 observed within-conversation repeated evidence occurrences. Forty-seven restored at shifted positions. These are evidence-chunk counts. The 1,220 loads above count complete KV pages.

| Measured path for the 40 natural turns | Native payload | Complete pages used in the replay |
| --- | ---: | ---: |
| P GPU → LMCache host store | 6.73 GiB | 6.35 GiB |
| LMCache host → P GPU restore | 2.91 GiB | 2.68 GiB |
| P GPU → D GPU via NIXL | 9.71 GiB | Outside the host cache replay |

Complete pages represent 94.3% of native store bytes and 92.2% of restore bytes. The remainder consists of boundary fragments. The native transfer comparison uses the MT-RAG workload label, excluding controlled probes. [Representative live Blend summary](https://github.com/Kibnelson/evidence_store_1/blob/a7b976f/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30568-20261123T021131Z/1p3d/blend/legacy-blend-summary.json)

## Exact packed replay on the five-turn capture

I selected `1534a095` for the complete byte-level codec pass because it was the smallest archive. The pass checked the canonical bytes of all 105 BF16 pages and tested 56 forward candidate pairs and their reverse direction. This capture contains 213 observed loads. The four packed policies use the same packed exponent format, minimum 5% per-frame delta saving, and one-hop reconstruction. [Published five-turn capture](https://github.com/Kibnelson/evidence_store_1/tree/a7b976f/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30568-20261123T021621Z/1p3d/blend) · [Packed replay implementation](../../scripts/vllm_pd_legacy_packed_replay.py)

| Policy | First tested budget for 213 hits | Hits at 124 MiB | Delta hits at 124 MiB | Rewrite bytes at 124 MiB |
| --- | ---: | ---: | ---: | ---: |
| Raw independent | 184 MiB | 0 | 0 | 0 |
| Original BF16 independent codec | 132 MiB | 207 | 0 | 0 |
| Packed BF16 independent codec | 128 MiB | 207 | 0 | 0 |
| Packed + forward base pinning | 124 MiB | 213 | 0 | 0 |
| Packed + demand-selected anchors | 124 MiB | 213 | 0 | 10.61 MiB |
| Packed + forward repacking on base eviction | 124 MiB | 213 | 0 | 14.06 MiB |

The extra six hits came from the graph's effect on which frames fit in cache. All 213 hits used standalone frames. I checked the reconstructed bytes and rewrites at each packed policy's first full-hit budget and at the common 128-MiB budget. The forward pinning policy matched the other graph policies' hit counts at 124 MiB with zero rewrites. Demand-selected anchors required nine rewrites and 14.06 MiB of associated encoded reads. Forward repacking required nine rewrites and 24.67 MiB of associated encoded reads.

For all 105 pages encoded independently, original compressed frames plus version IDs used 171.42 MiB, and packed compressed frames used 163.64 MiB. Bit packing saved 4.54% on top of the original independent codec. The differential graph lowered the tested full-hit capacity another 4 MiB, or 3.1% relative to packed independent storage in this conversation.

### CPU cost of the codecs

| Operation | Median CPU time per page |
| --- | ---: |
| Original independent encode | 36.5 ms |
| Packed independent encode | 93.9 ms |
| Original independent decode | 8.5 ms |
| Packed independent decode | 10.9 ms |
| Packed independent decode for the nine viable delta targets | 10.6 ms |
| Cold encoded base decode plus delta reconstruction for the same nine targets | 29.0 ms |

The first four medians use all 105 pages. The last two compare the same nine target pages whose best earlier-base delta met the 5% size threshold. The measured one-hop restore took approximately **2.7×** the CPU time of decoding the target's standalone packed frame. These are Python codec CPU times on the local host with encoded data already in memory. Archive reads, SHA verification, GPU staging, and request latency are outside the timing intervals. The 1.80-ms shadow apply above operates on an already-raw base and uses a different codec.

Across the six conversations, the trace-size replay shows a consistent storage benefit from independently compressed pages and weak savings from the traced exact-row format. In the five-turn byte-level replay, a one-hop graph bought a small extra retention gain and a hypothetical cold delta restore cost substantially more CPU. The packed-codec capacity and timing results cover this one conversation. The remaining five complete archives are published for a wider byte-level test. [Replay runbook](../vllm-legacy-full-blend-runbook.md#capture-the-six-complete-conversations-in-one-allocation)

## Faster exact restore paths

The follow-up benchmark used the same nine viable anchor–target pairs from `1534a095` for every path. All nine selected patches were XOR frames. It took the median of seven CPU timings per target and then the median across the nine targets. Every restore was checked byte for byte. The input frames were already in host memory. An encoded base includes base decompression in the timed interval. A decoded base is ready to use when timing starts.

| Exact codec and restore path | Standalone target | Delta with encoded base | Delta with decoded base |
| --- | ---: | ---: | ---: |
| Current packed frames with Python assembly | 10.49 ms | 28.75 ms | 18.20 ms |
| Same packed frames with native field assembly | 3.47 ms | 7.72 ms | 4.41 ms |
| BF16 field streams with whole-page Zstd XOR | 1.89 ms | 4.15 ms | 2.43 ms |
| Zstd streams with a delta only on exponents | 1.81 ms | 3.02 ms | **1.43 ms** |

**Fuse XOR and page assembly.** Splitting BF16 into exponent and sign/fraction is a reversible rearrangement of bits. The native path decodes both sets of fields, XORs corresponding fields and assembles the final page once. It uses the same stored packed frames and reduces the measured encoded-base restore from 28.75 ms to 7.72 ms. Applying native assembly to the standalone path also speeds it up to 3.47 ms.

**Depend on fewer base bytes.** The exponent-only path stores the target's sign/fraction stream independently. An encoded-base hit decodes only the anchor's exponent stream. A decoded exponent stream brings the matched target restore to 1.43 ms versus 1.81 ms for decoding both standalone target streams. The nine target frames occupy 12.27 MiB as exponent deltas versus 14.57 MiB standalone, a 15.8% reduction including frame and version metadata. Reading the nine compressed anchor exponent streams adds 4.44 MiB if every load starts with an encoded base.

**Charge the cost of decoded anchors.** The nine whole-page packed deltas save 3.45 MiB on their targets. Keeping those nine bases raw in place of their compressed copies would add 6.19 MiB. A decoded anchor already available from a prior load or staging batch can be shared by dependents during that batch. Long-lived decoded anchors need their own cache budget and reuse to justify that space.

On the same nine targets, the median standalone encode took 92.77 ms with the current packed codec, 7.46 ms with whole-page Zstd, and 6.82 ms with two Zstd field streams. These timings include the BF16 field transform and compression. They exclude candidate search and delta encoding.

At the conversation level, the whole-page Zstd prototype used 169.49 MiB for all 105 independent frames versus 163.64 MiB for the current packed codec, including version IDs. With Zstd, all 213 observed loads fit at the first tested budget of 132 MiB for standalone storage and 128 MiB with forward pinning. The pinned Zstd run verified one delta hit at 128 MiB and performed zero rewrites. The current packed graph reached full hits at 124 MiB. [Pinned capture and KV payloads](https://github.com/Kibnelson/evidence_store_1/tree/a7b976f/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30568-20261123T021621Z/1p3d/blend)

## Exponent-only store cost and cache coverage

The write benchmark used the same nine target and base pairs. It timed seven writes per path and verified the decoded output after each timing. Candidate lookup and storage I/O are outside the measured CPU interval.

| Zstd field format | Median CPU time to encode a page | Median CPU time to restore a page |
| --- | ---: | ---: |
| Standalone target | 6.91 ms | 1.81 ms |
| Exponent delta with decoded base exponent ready | 6.97 ms | 1.43 ms |
| Exponent delta with stored compressed base exponent | 8.56 ms | 3.02 ms |

If a full raw base page is already available at write time, extracting its exponent and encoding the target delta took 7.21 ms. For restore, the nine target frames require 12.27 MiB of reads if the decoded bases are ready versus 14.57 MiB for the matched standalone frames. With nine encoded bases, the exponent dependency adds 4.44 MiB and brings modeled host input to 16.72 MiB. Each restore produces the original 2.25 MiB GPU page.

I replayed all 105 stores and 213 load events using the exponent-only option, the same 5% admission threshold, and one-hop forward base pinning. Refill on a miss uses the original captured page. I verified the output bytes of every hit at each policy's first full-hit budget.

| Cache at 128 MiB | Local hits | Capacity misses | Hits using a delta | Rewrites |
| --- | ---: | ---: | ---: | ---: |
| Standalone Zstd field frames | 207 | 6 | 0 | 0 |
| Exponent-only delta option with forward pinning | 213 | 0 | 0 | 0 |

Standalone first reached 213 hits at 132 MiB on the 4-MiB grid. The graph reached it at 128 MiB by using smaller representations for some resident pages and keeping enough standalone target pages for all 213 loads. At their respective full-hit budgets, both policies read 343.7 MiB of encoded target frames. Retaining more page versions can preserve conversation history in a fixed host budget. The replay counts the avoided fallback events. A live run would determine whether each avoided fallback removes a slower-tier fetch, prefill recomputation, or both.

The six misses were one eviction cascade in the standalone replay. During turn 4, the cache held 127.54 MiB when a 1.61 MiB store pushed it over the 128 MiB limit. LRU evicted the turn-2 version at `[896, 912)`. Turn 5 needed that page again, and refilling it evicted the next needed page at `[912, 928)`. This repeated until six consecutive turn-2 pages covering `[896, 992)` had missed. The graph still held all six as standalone frames when turn 5 requested them. This cascade depends on chronological LRU and immediate refill after each miss.

The P → D benefit uses the same idea of reconstructing exact KV from a compatible base at a different location. D can receive less data if it already has a compatible base and the handoff uses a delta. The earlier changed-row handoff sent 35.18% fewer NIXL bytes across 156 measured requests. The exponent-only experiment models P-side host storage and restores complete GPU pages. [Earlier live changed-row handoff](../vllm-legacy-full-blend-tracing.md#smoke-check-on-archived-gpu-evidence)
