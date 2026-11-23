# Differential KV storage on prefill workers

## Summary

EvidenceStore could let a prefill worker retain more useful KV versions under the same cache capacity. When a later turn creates a new KV page, the store would choose between independent lossless compression, references to matching rows, and an exact delta against one or more earlier pages. The choice would account for the total space occupied by dependencies and the cost of reading and reconstructing a page.

Our MT-RAG trace shows why this is worth examining. The same **50 pages from turn 1** served all three later turns. A separate pair of pages from the growing conversation history covered the same 16 token IDs and had different KV bytes. Storing the newer page as a delta reduced its encoded size from **1.71 MB to 1.10 MB**. Across all **91 captured pages**, the best tested differential graph saved **3.00 MB (1.9%)** beyond the best independent lossless compression. This is an offline result from one conversation. The team decision is whether the extra saving survives realistic residency and restore costs. [MT-RAG raw capture][capture] · [Page report][page-report]

## The gap in the current path

P prepares prompt KV and D uses it to generate the answer. In the measured Full Blend configuration, P can restore a matching LMCache object and use that KV as input to Blend. A later turn can append conversation text or change the evidence. When that produces a new object for the cache, the measured store writes the full KV for that object. The earlier and later versions can still share information, including identical rows or predictable fields within changed rows. [Measured path and trace semantics][tracing]

Consider a conversation section that grows by one question. A page from the old section and a page from the new section can have the same token IDs and positions. KV is the output of model computation, so the token match is a way to **find** a candidate base. The actual tensor bytes determine whether we can reference rows, encode a compact difference, or store the new version independently. The cache needs a representation that captures that relationship and can reconstruct the exact bytes later.

This extends the original differential P → D idea. For a transfer, the question is whether the selected D can reconstruct the target using a base it already has or can cheaply fetch. For storage, the question is whether P's cache can represent the useful versions with fewer resident bytes while meeting the restore latency budget. This note focuses on the P-side cache. The archived run used a full NIXL handoff. [P → D delta model][handoff] · [Run configuration][provenance]

## Evidence from the published run

Job **30547** replayed four natural turns of one MT-RAG CLAPNQ conversation using `previous-turn-first`, one Blend-enabled P worker and three D workers. P restored KV for all **16 repeated evidence occurrences** out of **24 total occurrences**. We reviewed **91 complete pages written by P to LMCache**, with **16 tokens, 36 layers, and 1,152 K/V rows** per page. A row is the key or value for one token at one layer across the KV heads. Each row used **2,048 B**, giving **2,359,296 B** per raw page. The complete writes totaled **214,695,936 B**. LMCache object boundaries can cut across pages. The trace recorded another **29,196,288 B** of boundary writes separately from the complete-page totals below. [Run configuration][provenance] · [Replay summary][replay-summary] · [Raw capture inventory][raw-inventory] · [Movement accounting][movement]

| Question | Finding |
| --- | --- |
| Which pages did later turns actually load? | The same 50 turn-1 pages each served turns 2, 3, and 4. Seven turn-3 pages each served turn 4. This gives 157 observed complete-page loads from 57 stored versions. |
| Can exact rows come from different bases? | Turn 3's page at `[896, 912)` used 18 of its 32 layer-0 rows from two earlier pages, compared with 12 from the best single base. The offline proof reconstructed all 36 layers byte for byte. |
| How often did moving the source slot expose exact reuse? | Allowing a source row at another token slot added 570 matches in the capture-wide search. All 570 were layer-0 V rows. The search includes earlier writes from the same turn. Restricting it to earlier turns added 150 matches. These match counts precede reference, occupancy, and read costs. |
| Is there structure in changed rows? | In ten aligned earlier-turn base–target pairs, BF16 high bytes matched 76% for K and 67% for V. Low bytes matched 8% and 5%. The high byte contains the sign and most exponent bits. This is a signal for conditional lossless encoding. |
| Did the previous layer make a good base? | A within-page check found zero exact K/V row matches across layers. Writing each layer as an XOR against its predecessor before byte-shuffled zlib used 4.4% more bytes than compressing those layers independently. |

The load counts join each native restore to its measured source store on the recorded page version, tensor digest, and request identity. The native shadow codec verified one changed-page reconstruction at an actual later load. The serving path supplied the raw page while the trace independently verified its shadow reconstruction. Slot remapping, alternate bases, and compressed encodings were evaluated offline on the captured tensors. [Native loads and shadow verification][lookup] · [Captured tensors][payloads]

### Two pages that motivate the design

**Exact row references within one layer.** Turn 3's evidence page at `[896, 912)` can draw rows from turn 1's `[64, 80)` page and turn 2's `[896, 912)` page. The turn-1 source was directly loaded on all three later turns. The turn-2 source had zero direct loads in the recorded window, so using it creates an extra retention decision. At layer 0, the modeled representation was **28,864 B** with two bases versus **41,082 B** with one, including the row selectors and base metadata. The turn-2 and turn-3 pages at the same token position belong to different chunks of the same source document. When turn 4 loaded the target, the live trace verified a simpler shadow reconstruction using turn 1's page at `[896, 912)`. The two-base reconstruction uses the same target tensor and was verified offline. [Turn-4 load and reconstruction digest][row-target] · [Chunk spans and document IDs][chunks] · [Turn-1 base tensor][row-base-first] · [Turn-2 base tensor][row-base-second] · [Target tensor][row-target-payload]

**Partial similarity in conversation history.** Turns 3 and 4 kept all six evidence chunks in the same token spans. The strongest compressed delta used the page at `[1056, 1072)` in the conversation section after those chunks. The archive records an 88-token common prefix in that LMCache object and 16 matching token IDs in the page. The target still had different KV bytes. Independent lossless encoding used **1,713,611 B**, and the best tested encoded delta used **1,102,114 B**. The base used **1,713,612 B** independently compressed and had zero direct loads in the trace. The target was written on the final turn, so this example supplies an offline storage and reconstruction result. [Page pair and observed token overlap][partial-target] · [Prompt construction][renderer] · [Base tensor][partial-base-payload] · [Target tensor][partial-target-payload]

## How large is the opportunity?

| Representation for the 91 captured pages | Stored bytes | Saving vs. raw |
| --- | ---: | ---: |
| Native raw tensor bytes | 214,695,936 | — |
| Independent plain zlib level 1 | 174,420,545 | 18.8% |
| Best tested independent encoding, BF16 exponent and sign-and-fraction streams compressed separately | 155,737,940 | 27.5% |
| Best tested differential graph, whole-page lossless residual against an earlier turn | 152,736,860 | 28.9% |

The differential graph used **eight deltas and eight independent bases**, selected from **39 candidate pairs**. Its matched independent encoding baseline was **155,801,144 B**, giving **3,064,284 B** of savings for that comparison. Comparing the best tested result in each class gives **3,001,080 B**, or **1.9%**, beyond independent compression. These are encoded tensor totals with modeled representation metadata. The graph charges for the selected bases as part of the total. The published trace provides the raw inputs and actual native loads. [Archive and raw inputs][capture]

Exact-row references address a different kind of redundancy. One earlier base per target with layer-0 slot remapping saved **1,211,521 B** relative to raw storage after reserving the needed source layers. Greedy selection of up to two bases fit some pages better while reserving more source layers, for a total saving of **1,083,630 B** in the same search. The rows and bit-level totals overlap because both compare representations of the same captured pages. Each target contributes one selected representation to a storage budget.

**The impact we need to establish is useful cache capacity.** A smaller representation matters if it keeps more directly reusable KV or reduces evictions and prefill recomputation under a fixed budget. The largest individual page reduction was about **36%** versus its independent encoding. The gain across this conversation was about **1.9%** beyond independent compression. The next evaluation needs to show how often that incremental gain is available and what it costs to restore.

## Proposed storage policy

Give each produced page version an immutable identity. Track its model revision, cache layout, dtype, layer and K/V geometry, logical span, and content digest. Use conversation lineage, token alignment, chunk and document IDs, and inexpensive tensor signatures to propose compatible earlier bases. A match in the metadata ranks a candidate. Checks against the target bytes decide which representation is valid.

| Signal in the data | Candidate representation |
| --- | --- |
| An entire tensor or region is byte-identical | A reference to the existing data |
| Exact rows are scattered across slots | A map from source slots to target slots, plus the replacement rows |
| An earlier tensor predicts parts of the new tensor | Compare lossless encodings such as XOR residuals, modular integer differences on BF16 fields, and entropy coding conditioned on the base |
| Different regions have different good sources | Let each region choose a base and charge all references and dependencies |
| Independent compression has the lowest cost | Store a self-contained compressed version, with raw bytes as a fallback |

Start with a bounded graph in one P cache. A stored version can depend on an earlier compatible base. A depth-one representation uses independently encoded bases, which makes their lifetime and reconstruction costs easy to inspect. Each dependent records its base version, source selection, format, and checksum. Commit the dependency with the dependent. If a base is selected for eviction, first materialize or evict its dependent versions according to the cache policy. On a cache hit, reconstruct the original LMCache tensor bytes before the existing GPU staging and Blend position adjustment.

### Account for a base over its lifetime

A fair comparison counts each retained base once, at the size and lifetime the graph actually requires. A page already retained for direct hits adds little or no incremental storage if it can serve as a base in that representation. A page kept solely for one or more deltas consumes cache capacity until the last dependency is released.

```text
Incremental storage saving =
    independently encoded target bytes
    − target patch bytes, including metadata
    − extra bytes required to retain or promote its base
```

For the turn-4 conversation example, if both versions are retained, the target delta saves **611,497 B**. If the older version would otherwise be evicted and has no other dependents, keeping it for this target costs **2,815,726 B** together with the patch. Storing just the target independently costs **1,713,611 B**. A cold restore that reads the entire encoded base and the patch also reads **2,815,726 B** of encoded input for that target. A restore with the base already decoded has a different cost. Measure base lookup, bytes read, decoding, materialization, and staging for each path. [Page pair][partial-target]

## Test plan

### 1. Reproduce the evidence on CPU from the existing archive

Use job 30547 as a fixed input. Verify the archive, then join page writes and later native loads by page ID, version, and digest. Reproduce the two examples and record candidate selection, exact row or field coverage, encoded target bytes, source version IDs, and reconstruction hashes.

The archived comparator runs the same-slot row-replacement policies and reports both gross totals and totals after reserving required raw bases. It reproduced **722,332 B** of savings versus raw for the best per-layer policy after reserving its bases. The slot-remapping and byte-level figures in this note are from exploratory offline comparisons on the same inputs. Record their output in a reusable report as part of this step. [Pinned archive verifier][verifier] · [Pinned comparator][comparator]

```bash
archive=reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z

python3 scripts/vllm_pd_legacy_blend_archive.py --verify "$archive"
python3 scripts/vllm_pd_legacy_page_compare.py "$archive/1p3d/blend" > /tmp/kv-page-30547-comparison.json
```

For the compression comparison, read each captured tensor as `[layer][K then V][token][row bytes]` and verify its SHA-256. Compare the raw layout, byte shuffle, BF16 exponent separation, and bit planes, using the same lossless compressor and level for independent and residual encodings. For each target, select the most recent compatible page from an earlier turn for each of three candidate rules. Match the logical page ID, the absolute start token, or the start offset within the LMCache object. Add bases identified by the trace and deduplicate the candidates. The whole-page trial charges 48 B of framing and 64 B per base descriptor. The two separately compressed BF16 streams add a 4 B length field per stream. Restore selected pages and check every byte against the capture. [Canonical payload reader][payload-reader] · [Page trace records][page-report]

### 2. Replay retention under the same cache budget

Replay the observed stores and loads in timestamp order. Compare native raw storage, the strongest self-contained lossless encoding, and a differential graph with the same cache budget and request order. Use only the pages available at the time of each decision. The observed next-use information is useful as an upper bound when evaluating policies.

The packed replay compares the original codec, compact row references, and compact row references combined with an exact BF16 exponent palette. It runs independent compression, forward pinning, demand-selected anchors, and materializing dependents before eviction. The tool accepts multiple raw capture directories and starts a fresh logical cache for each one. The full-capacity comparison uses committed encoded frames. Capture scratch, reconstruction buffers, and compression time have separate costs to measure before serving.

```bash
archive=reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z
python3 scripts/vllm_pd_legacy_packed_replay.py "$archive/1p3d/blend" \
  --capacity-mib 96 97 98 99 100 101 102 103 104 105 > /tmp/packed-kv-replay.json
```

To replay more conversations, pass their published `1p3d/blend` directories before the capacity options. Use the [conversation selector in the Full Blend runbook](vllm-legacy-full-blend-runbook.md#trace-differential-storage-on-mt-rag) to capture one of the six validated CLAPNQ conversations on Slurm. Start with an inclusive window of turns 1–4 and check that the report has all complete pages before including the capture. The offline replay also accepts fractional MiB capacity points near a threshold.

In job 30547, compact row references alone had the same full-hit capacity as the original format. Packing all 91 pages' exponent streams with 4-bit codes and exact escape bytes reduced the total independent frame size including version IDs from **155,741,701 B to 148,668,515 B**. On the 1/16-MiB grid, independent compression, demand-selected anchors, and forward repacking each served all **157** observed loads at **98.1875 MiB**, down from **102.875 MiB**. At that packed budget all 157 loads came from independently encoded pages. The Python palette prototype took **97.1 ms** to encode and **11.0 ms** to decode at the median of a matched four-page CPU sample, versus **37.4 ms** and **8.8 ms** for the original encoding. The cache capacity and codec timing results measure different parts of the design.

- Track current and peak resident bytes, dependency metadata, base pin time, and unique bytes read. Include boundary fragments in the total cache budget at their baseline representation.
- Measure target hits, base hits, evictions, fallbacks, and which later loads each policy would preserve. Test a warm decoded base, a stored compressed base, and an evicted base.
- Compare one-base and two-base plans under the same accounting. Measure the best per-page plans and the graph after their dependencies have been reserved.
- Advance only when demanded pages reconstruct byte for byte, the graph saves bytes relative to independent compression after retention and metadata, and the restore cost can be explained at the same capacity.

### 3. Check generalization before changing the serving path

Use additional MT-RAG conversations with a range of lengths, repeated evidence, appended or edited history, and retrieval changes. Keep conversation boundaries when separating any data used to tune candidate ranking or conditional coding from the data used to evaluate it. Keep the pinned model, retrieval setup, and prompt policy matched within each comparison. Report results separately for evidence and conversation regions and distinguish ordinary cache hits from reconstruction hits. The earlier 40-turn storage trace provides additional demand and object-level evidence. The codec comparison requires complete raw page captures. [Earlier MT-RAG object-level trace][earlier-storage]

### 4. If the offline results justify it, run a gated P-side smoke test

Start with the existing one-Blend-worker and three-decoder setup. Compare native raw, independent compression, and differential storage with a fixed P cache budget. Keep the same inputs and full NIXL handoff for those runs. Capture candidate lookup, store encoding, base lookup and read, reconstruction, GPU staging, prefill recomputation, and request time to first token. Report p50 and p95 restore and request timings, along with resident bytes and eviction counts. Use the existing trace gate and a timing run with detailed tracing disabled to check instrumentation overhead. Repeat with two Blend workers and two decoders once single-P behavior is understood, so routing and base locality can be measured. [Existing page capture runbook][runbook]

The team review should decide which capacity range and restore latency budget to use, when a base can be pinned, and which evidence would justify an online implementation. The first deliverable can be a reproducible offline report from the existing capture showing **which representations win, which bases they keep, and how many of those pages were subsequently needed**.

[capture]: https://github.com/Kibnelson/evidence_store_1/tree/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z
[page-report]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z/1p3d/blend/legacy-kv-page-report.json
[provenance]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z/1p3d/blend/provenance.json
[replay-summary]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z/1p3d/blend/legacy-blend-summary.json#L66-L109
[lookup]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z/1p3d/blend/legacy-kv-page-report.json#L2141
[movement]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z/1p3d/blend/legacy-kv-page-report.json#L2185
[raw-inventory]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z/1p3d/blend/legacy-kv-page-report.json#L2210
[payloads]: https://github.com/Kibnelson/evidence_store_1/tree/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z/1p3d/blend/page-payloads
[chunks]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z/1p3d/blend/legacy-chunks.jsonl.gz
[row-target]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z/1p3d/blend/legacy-kv-page-report.json#L12039-L12066
[row-base-first]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z/1p3d/blend/page-payloads/2c8e5b9bd74d8f987fb24a8742e4dfec96fbb77811c3886ad5b3f2c8e2a9396e.bin.gz
[row-base-second]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z/1p3d/blend/page-payloads/12c03648f9b9b63ca34ce215514dee5491f4c9c42969bc1ec69318ce89731111.bin.gz
[row-target-payload]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z/1p3d/blend/page-payloads/92b6f46f2c6ade7772ce0e859b1b496b7ab6d5b8bf542eef9cb25b815b762f3a.bin.gz
[partial-target]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z/1p3d/blend/legacy-kv-page-report.json#L20379-L20396
[partial-base-payload]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z/1p3d/blend/page-payloads/04dc4d84dfe184e219e828b084bf1604c3a8e3eb614e264709e3b7bf08bfaaa0.bin.gz
[partial-target-payload]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30547-20261122T210618Z/1p3d/blend/page-payloads/ad709fabdb0507c01fa75fea0eae311e913ea1b40904febc102c2f36647fc76d.bin.gz
[renderer]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/src/evidencestore/retrieval.py#L485-L492
[tracing]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/docs/vllm-legacy-full-blend-tracing.md#differential-storage-on-mt-rag
[handoff]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/docs/vllm-legacy-full-blend-tracing.md#estimate-a-changed-row-nixl-handoff
[verifier]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/scripts/vllm_pd_legacy_blend_archive.py#L268
[comparator]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/scripts/vllm_pd_legacy_page_compare.py
[payload-reader]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/scripts/vllm_pd_legacy_page_payloads.py
[earlier-storage]: https://github.com/Kibnelson/evidence_store_1/tree/b3b7e0a/reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30529-20261122T162556Z
[runbook]: https://github.com/Kibnelson/evidence_store_1/blob/f4b0471/docs/vllm-legacy-full-blend-runbook.md#trace-differential-storage-on-mt-rag
