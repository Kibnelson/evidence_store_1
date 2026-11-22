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

## Differential storage on MT-RAG

`--trace-kv-storage` samples KV objects committed by the in-process LMCache CPU backend during Full Blend prefill. It samples the first, middle, and last layer and records the LMCache key digest, token span, tensor version, and raw byte count. LMCache storage objects can have different boundaries from the vLLM blocks tracked by NIXL. [Run the MT-RAG trace](vllm-legacy-full-blend-runbook.md#trace-differential-storage-on-mt-rag)

For each eligible object, the trace checks up to four recent resident raw bases with matching layout and a shared token prefix. It compares actual K and V bytes and charges the full replacement tiles covering appended tokens. It records the smallest exact replacement patch it found when the patch and its manifest fit in fewer bytes than the raw tensor. Both representations use the same native position metadata. The reported savings cover sampled tensor payloads and patch manifests. Additional base retention costs need a separate measurement.

When LMCache serves a sampled target, the trace checks that the base is still resident, applies the patch in a shadow buffer, and verifies the result against the tensor bytes returned by the native load. It records base and patch input bytes, codec time, and the time of the full diagnostic check. The report distinguishes observed patch opportunities from targets that were loaded and reconstructed. Events contain byte counts and digests. [Storage trace code](../scripts/vllm_pd_legacy_storage_trace.py)

### Native page smoke

`--trace-kv-pages` follows turns 1–4 of CLAPNQ conversation `1534a095279f2cb888fb0bea17bd70da` on P0. The match uses the native slot mapping. A complete page covers 16 consecutive slots in one physical block and lies wholly inside an LMCache object. The report retains boundary bytes separately. It uses the logical page version to join a store to a later load, and records both physical block locations so a shifted restore can span multiple destination blocks.

For each full page, the trace compares K and V across all model layers against up to two compatible resident raw bases at the same offset within another LMCache object. Candidate objects share at least eight starting tokens. The report records the selected base and target digests, source spans, changed rows by layer, raw and encoded bytes, and which later requests read the target. The reconstruction check reads the CPU base and applies one patch before Blend position adjustment. The report distinguishes byte-exact proofs, base availability, and pages with no observed load. It records sampling limits, including the bounded shadow patch memory. [Run the page smoke](vllm-legacy-full-blend-runbook.md#trace-differential-storage-on-mt-rag)

The optional `--capture-kv-pages` stores the exact full tensor from each measured LMCache put as a gzip file named by its SHA-256. Its layout is `[layer][K or V][token slot][row bytes]`. A later restore points to the stored version, so one file can support multiple reads. Publication follows the natural-turn records and verifies each file against the reported uncompressed length and tensor digest. The codec is framing for the offline analysis. The gzip size measures the capture file and is independent of the proposed KV representation size.

For offline base search, the comparison script considers earlier captured compatible pages. The same tensors feed the one-base and per-layer alternatives. It reports representation size and base dependencies. The native trace still supplies store and restore events for assessing observed demand and availability. [CPU-only comparison](../scripts/vllm_pd_legacy_page_compare.py)

## Estimate a changed-row NIXL handoff

The earlier GPU-only run retained the previous complete prompt KV on the same D for each conversation. B compared fingerprints of the final KV rows with that base. D reused matching rows and read complete replacement rows for changed or new data over NIXL. It then verified the reconstructed target before decoding. Job 29530 sent **116,535,052,288 of 179,608,780,800 logical target bytes** across its measured turns, a **35.12% payload reduction** under that retention policy. [Earlier run card](https://github.com/Kibnelson/context_pilot_paper/blob/e8cb79f3d668a795c44e2d878b2baf2c47444ad2/docs/evidencestore/implementation/scheduler_v2_phase2_native_and_three_modes_run_card.md#what-differs-between-the-runnable-paths) · [Job 29530 report](https://github.com/Kibnelson/context_pilot_paper/blob/f8d8da26b19f70f0e37beb4b85bc3f91463ad4ca/traces/scheduler_v2_phase1_fullpath_diagnostic/dd02d619c0ef_blend_delta_live_job29530/blend_xor_review/publication/report.md)

### Calculate actual bytes from the final Blend output

Blend changes the candidate rows in two ways before the final KV is ready.

1. **Restore and adjust positions.** LMCache loads cached K and V on P. With cached positions enabled, it applies rotary position adjustment to the restored **K** at the destination offsets. The adjustment leaves **V** as it was in the restored buffer. [Position adjustment in the pinned LMCache connector](https://github.com/LMCache/LMCache/blob/389b9cfcb5cec502dbba6b5d13725bf2e024610d/lmcache/v1/gpu_connector/gpu_connectors.py#L881)
2. **Select and repair.** At the check layer, Blend scores each active token using `‖K_fresh − K_restored_at_current_position‖²`. This run selects the top 15% of that set, rounded down with a minimum of one. For selected positions it writes fresh **K and V** into the restored buffers, then continues with those positions in later layers. vLLM also writes KV for positions assigned to ordinary prefill. [Blend scoring and replacement](https://github.com/LMCache/LMCache/blob/389b9cfcb5cec502dbba6b5d13725bf2e024610d/lmcache/v1/compute/blend/blender.py#L89) · [Run settings](../scripts/vllm_pd_cluster.py#L685)

Blend's score compares two K values on **P** to choose work. The transfer comparison happens later and compares **final P rows with the selected D's retained rows**. A shifted K can change without selective repair. An unselected V can differ from D's row if LMCache loaded it from a different base. A repaired row can still match D's base byte for byte. The current trace counts positions that reach attention and MLP. Measuring actual row changes also requires fingerprints from the last writes, including any ordinary prefill writes. [Current Blend trace hook](../scripts/vllm_pd_legacy_trace_overlay.py#L159) · [Earlier row comparison code](https://github.com/Kibnelson/context_pilot_paper/blob/e8cb79f3d668a795c44e2d878b2baf2c47444ad2/scheduler/v2/phase1_blend_delta_chunks.py#L35)

Match each target row `r` to the decoder base by segment identity and occurrence, token offset within the segment, layer, and K/V side. That gives the following three cases.

| Final row on P | Action on D |
| --- | --- |
| Compatible retained row with the same final fingerprint (`U`) | Reuse D's row at the target position. |
| Compatible retained row with a different final fingerprint (`C`) | Read the complete replacement row over NIXL. |
| No compatible retained row (`N`) | Read the complete current row over NIXL. |

For each set, sum the byte size `b(r)` of its rows.

```text
unchanged bytes U_b     = sum of b(r) over r in U
changed bytes C_b       = sum of b(r) over r in C
missing-base bytes N_b  = sum of b(r) over r in N
logical full KV bytes   = U_b + C_b + N_b
logical delta KV bytes  = C_b + N_b
logical KV reduction    = U_b / (U_b + C_b + N_b)
```

For example, consider one layer with two tokens from a restored chunk and one token from a new chunk. Suppose both restored tokens have a compatible base on D and the final comparison gives these results.

| Token at this layer | Final K compared with D | Final V compared with D | Rows sent |
| --- | --- | --- | ---: |
| Restored token at a shifted position, outside the repair selection | Different after position adjustment | Exact match | 1 |
| Restored token selected for repair | Different | Different | 2 |
| Token from the new chunk | No base | No base | 2 |

The example has six equal-size rows. D reuses one, receives three replacements, and receives two rows without a base. Sending five of six avoids **16.7%** of logical payload before transfer layout and overhead. Apply the same comparison independently at every layer.

Report the actual packed NIXL payload and manifest bytes separately. The native full READ is block-rounded, so compare NIXL completion counters for the transport reduction and use the logical equation to explain reuse. Verify the reconstructed target against P's full digest before D decodes. A request with no usable retained base takes the full path. Keep the LMCache restore and Blend work receipts to attribute reuse and recomputation to chunks. Those receipts provide context for the row comparison.

### Measure the speed on a matched replay

Start with MT-RAG. Keep each conversation on one D and retain the previous completed turn, as job 29530 did. Compare a forced full READ and a changed-row READ on the same requests, prompt policy, Blend settings, topology, route, retention budget, and arrival schedule. MuSiQue's independent questions need a compatible cross-request base on the selected D to test the same mechanism.

Measure NIXL submission → completion, P KV ready → D target ready for decode, and client TTFT in both arms. Keep P HTTP time, fingerprinting and packing times, D base lookup and reconstruction time, and throughput over the measured replay window. Include requests that fall back to a full READ. A zero-payload handoff has zero READ time and still contributes to handoff time and TTFT. For example, median handoff speedup is full-mode p50 handoff time divided by changed-row-mode p50 handoff time. Compare output-token throughput over the same workload.

In the earlier 20-conversation comparison, mean observed READ time went from **34.317 s to 7.480 s** and mean TTFT from **48.066 s to 20.131 s**. That comparison also changed routing and decoder retention. The next matched replay keeps those settings fixed and checks target reconstruction and generated output separately. [Prior full Blend run](https://github.com/Kibnelson/context_pilot_paper/blob/e390b7a0cad204c14f67c39439ebf208baf4df7c/traces/scheduler_v2_phase1_fullpath_diagnostic/e25c6fbba107_blend_20261025T202746Z/blend_xor_review/publication/report.md) · [Prior changed-row run](https://github.com/Kibnelson/context_pilot_paper/blob/f8d8da26b19f70f0e37beb4b85bc3f91463ad4ca/traces/scheduler_v2_phase1_fullpath_diagnostic/dd02d619c0ef_blend_delta_live_job29530/blend_xor_review/publication/report.md)

### Predict the speedup and break-even point

The [short formula sheet](kv-delta-speedup.md) gives the calculation and a worked example on one page.

Start with the final per-layer K and V comparison above. **Base coverage** tells us how much of the full target D can compare against its retained KV. The **diff ratio** tells us how much of that covered KV changed. The **NIXL sent ratio** is what enters the speed formula.

| Quantity | Formula | Meaning |
| --- | --- | --- |
| Logical full KV `L` | `U_b + C_b + N_b` | All final target KV bytes before transfer padding. |
| Base coverage `q` | `(U_b + C_b) / L` | Fraction of the target with a compatible row on D. |
| Diff ratio `d` | `C_b / (U_b + C_b)` | Fraction of the covered bytes whose final row differs. Count the full width of each changed row. |
| Logical sent ratio `r` | `(C_b + N_b) / L = 1 − q × (1 − d)` | Fraction of target KV that must travel, including rows without a base. |
| NIXL sent ratio `ρ` | `(L × r + O_delta) / (L + O_full)` | Fraction of full NIXL payload sent by the delta READ, including the layout of each transfer. |

`O_full` and `O_delta` are the extra bytes carried by their respective READs. Account for metadata sent separately in the time it adds to the critical path. When `q = 0`, `d` is undefined, `r = 1`, and the protocol uses the full path. For use with measured traces, `ρ = B_delta / F` directly, with `B_delta` the completed delta READ bytes and `F` the completed full READ bytes for the matching request.

```text
full NIXL bytes F          = L + O_full
packed delta bytes B_delta = C_b + N_b + O_delta
NIXL bytes saved           = F × (1 − ρ) = U_b + O_full − O_delta
```

For a latency `T` of interest, use `T_full` as the measured baseline. For **handoff time**, the clock runs from P KV ready to D ready to decode. **TTFT** adds client start → P KV ready and D ready → client first token to that handoff. Let `H` be the net additional time outside the READ on the chosen critical path, in seconds. Depending on the interval, it can include fingerprinting, base lookup or restore, packing, metadata delivery, reconstruction, and verification. Count overlapping work by its effect on elapsed time.

Use this first estimate when both modes issue a READ with the same effective payload bandwidth `W` (bytes/s) and fixed cost, and the READ is on the critical path.

```text
net time saved    = (F / W) × (1 − ρ) − H
predicted T_delta = T_full − net time saved

f = (F / W) / T_full     # fraction of baseline latency spent on byte-dependent READ time
h = H / T_full           # added critical-path cost as a fraction of baseline latency
predicted speedup = 1 / (1 − f × (1 − ρ) + h)
break-even ρ      = 1 − h / f
```

The delta improves latency when the saved bytes exceed `W × H`, or when `ρ` is below break-even. To work backward from a target speedup `S_goal`, solve for `ρ_max = 1 − (1 − 1/S_goal + h) / f`. At fixed base coverage, transfer layout overhead, and timing costs, that allows a diff ratio of at most `d_max = (ρ_max × F − N_b − O_delta) / (U_b + C_b)`. This inversion needs positive bandwidth-dependent time (`f > 0`) and compatible base rows. A negative `d_max` means even reusing every covered row falls short of the goal under this model.

For example, a **100 MB** target has **40 MB** of exact matches, **35 MB** of changed rows, and **25 MB** without a base after Blend. Coverage is **75%**, the diff ratio is **35 / 75 = 46.7%**, and the sent ratio is **60%** when both layouts add zero bytes. At **1 GB/s**, sending 60 MB saves **40 ms**. With a **110 ms** full handoff and **15 ms** of added handoff work, the delta takes **85 ms**, or **1.29×** faster. To achieve at least **1.25×** on that same handoff, the model requires `ρ ≤ 63%`, or `d ≤ 50.7%` at the same coverage. If full TTFT is **400 ms** and fingerprinting before P is ready adds another **5 ms**, the predicted TTFT is **380 ms**, or **1.05×** faster. This example uses decimal MB and GB.

For a calibrated model, fit `READ time = fixed cost + payload bytes / effective bandwidth` to positive-payload transfers at comparable decoder load. Fit the two modes separately when packing changes the descriptor layout or effective bandwidth. Then replace `(F − B_delta) / W` with the difference between the two predicted READ times. A zero-byte handoff with no READ has zero READ time and still pays its preparation and reconstruction costs. A request that falls back to full must include time already spent on the delta attempt and any bytes already moved. Use the portion of READ time exposed on the critical path when stages overlap. The trace records both the observed READ window and native NIXL duration for calibration. [NIXL completion telemetry](../scripts/vllm_pd_legacy_trace_events.py#L67)

On calibration requests, estimate `H` as `(T_delta − exposed READ time in delta mode) − (T_full − exposed READ time in full mode)`. On separate matched requests, make predictions from the row counts and those calibrated costs, then compare with observed handoff time and TTFT. Keep each request's `U_b`, `C_b`, `N_b`, `F`, `B_delta`, and latency intervals so prediction error can be checked across different coverage and diff ratios. For workload byte ratios, sum numerators and denominators before dividing. For latency, predict per request and compare matched summaries such as `p50(T_full) / p50(T_delta)`. Include every request in handoff and TTFT comparisons. Measure throughput directly as completed output tokens divided by replay wall time so concurrent requests contribute through their actual overlap.

### Smoke check on archived GPU evidence

The offline check rebuilt the published Full Blend and changed-row evidence for jobs 29538 and 29530 from their pinned Git objects. It verified all 971 input files and reproduced all 352 request records across the two modes byte for byte. For all 176 matched requests, the changed-row partition added to the full logical target, sent bytes equaled changed plus missing-base bytes, and D's recorded digest matched P's target. All 20 warm-ups sent the full target. The minimum-payload case, `shifted:005:00`, sent zero NIXL bytes and still took 829.5 ms from P ready to native import. [Full Blend evidence](https://github.com/Kibnelson/context_pilot_paper/blob/e390b7a0cad204c14f67c39439ebf208baf4df7c/traces/scheduler_v2_phase1_fullpath_diagnostic/e25c6fbba107_blend_20261025T202746Z/blend_xor_review/publication/report.md) · [Changed-row evidence](https://github.com/Kibnelson/context_pilot_paper/blob/f8d8da26b19f70f0e37beb4b85bc3f91463ad4ca/traces/scheduler_v2_phase1_fullpath_diagnostic/dd02d619c0ef_blend_delta_live_job29530/blend_xor_review/publication/report.md)

| Byte-weighted result over the 156 measured requests | Value |
| --- | ---: |
| Base coverage `q` | 98.59% |
| Diff ratio `d` among covered rows | 64.38% |
| Logical KV sent ratio `r` | 64.88% |
| NIXL sent ratio `ρ` including the native full READ's padding | 64.82% |

The prior concurrent runs have different routing and retention. A simple linear fit of READ time against bytes explains under 6% of timing variance in either arm. For the current serial CLAPNQ full-READ traces, a trial fit per policy using three conversations for calibration and three for validation gave effective rates of 0.436–0.448 GB/s and mean absolute READ prediction errors of 12.7–20.5 ms on the held-out turns. Those fits describe the full READ over the observed sizes. The packed delta READ and its critical-path overhead still need their own measurements. [Current CLAPNQ stage timings and archives](datasets/mtrag.md#measured-stage-timings)

### First EvidenceStore GPU smoke

Use 1B/3D as in the earlier one-conversation smoke. Freeze the prompts and generation settings for turns 1–3 of CLAPNQ conversation `1534a095279f2cb888fb0bea17bd70da` under `previous-turn-first`. In the current archive, turns 2 and 3 each restored five chunks found in the immediately preceding turn. Run the same three prompts against full READ and changed-row READ, with fresh state for each arm and the conversation assigned to the same D. [Earlier one-conversation fused smoke](https://github.com/Kibnelson/context_pilot_paper/blob/35a67d4f85ff5a7bd67f2367c11480f9a892de37/traces/scheduler_v2_phase1_fullpath_diagnostic/8803df941345_blend_delta_live_job29484/blend_xor_review/publication/report.md) · [Selected CLAPNQ chunk rows](../reports/smoke/mtrag-multi-turn-repeat-legacy-full-blend-30511-20261120T150130Z/1p3d/blend/legacy-chunks.jsonl.gz)

Check the cold full send first, then verify the retained predecessor and final digest on D for both later turns. Reconcile `U_b + C_b + N_b` with final target size and the packed plan with completed NIXL bytes. Record all timing boundaries for each arm and verify all three row outcomes across the smoke. Use a controlled prompt pair that shifts cached evidence if the natural turns leave a row outcome untested. Use more matched turns and payload sizes to calibrate speed after this correctness check passes.

Include **gateway → P prompt delivery** in both arms. For each request, check that the selected P logs receipt of the final HTTP body bytes before JSON parsing, that the events join by `X-Request-Id`, and that the gateway's serialized body length matches the count at P. Retain the prompt token count, HTTP body bytes, and body-write-start → body-received duration. Show that delivery interval alongside gateway preparation, the full P HTTP round trip, and the later P → D NIXL transfer. [Gateway delivery trace TODO](#follow-ups-from-the-end-to-end-trace-review)

The current EvidenceStore runtime uses full NIXL READs and removes the earlier final-write fingerprint hooks during its build. This GPU smoke needs those hooks and the changed-row decoder path integrated first. [Current standalone build patch](../third_party/legacy-full-blend-standalone.patch)

### Sensitivity estimates from the existing full-READ traces

The existing runs provide a full-payload baseline and a P-side candidate pool. Use these quantities to explore scenarios while final-row comparisons and decoder retention are pending.

| Symbol | Meaning |
| --- | --- |
| `F` | Measured full NIXL payload bytes, including the native block padding. |
| `R` | Candidate KV bytes within verified LMCache restores of repeat evidence on P. Count each restored destination range once. Leave the rest of `F` in the payload, including new evidence and prompt text outside the selected ranges. |
| `a` | Fraction of `R` for which the selected D has a compatible retained base row. |
| `c` | Fraction of those base-covered bytes that must still be sent. For packed replacements, count the full width of each changed row. Coarser transfer granularity can require sending extra bytes. |
| `M` | Extra bytes for the delta manifest and format, accounted separately from KV. |

```text
KV bytes that can be omitted = R × a × (1 − c)
estimated delta bytes        = F − R × a × (1 − c) + M
bytes sent with full fallback = min(F, estimated delta bytes)
payload reduction             = 1 − bytes sent / F
ideal bandwidth-bound speedup = F / bytes sent
```

When `a = 0`, or `c = 1`, the full transfer is the byte-minimizing choice. For a batch of requests, sum the byte counts before taking either ratio. The dataset tables assume packed rows and set `M = 0` to show sensitivity. A tiny numerical change in every row can still result in `c = 1` with complete row replacements. Blend token-layer visits measure work. Determining `c` requires comparing **final K and V rows** against D's base after position handling and any later writes. The current traces establish P-side restores; `a` and `c` remain scenario inputs. [Earlier row comparison code](https://github.com/Kibnelson/context_pilot_paper/blob/e8cb79f3d668a795c44e2d878b2baf2c47444ad2/scheduler/v2/phase1_blend_delta_chunks.py#L35)

Use the [latency model](#predict-the-speedup-and-break-even-point) with `ρ` equal to the scenario's estimated NIXL payload divided by `F`. Account for separately delivered metadata in `H`.

## Follow-ups from the end-to-end trace review

- [ ] **Measure gateway → P prompt delivery.** Under the existing trace gate, timestamp the start of the gateway's HTTP body write and the arrival of the final body bytes at P's HTTP API before JSON parsing. Join the events by `X-Request-Id`, reconcile sent and received body bytes, and retain the prompt token count, body length, and elapsed delivery time in the compact request archive. Use the resulting application-level delivery interval and byte count as a separate row alongside the existing P HTTP round trip. [Current gateway send and response events](../src/evidencestore/gateway.py#L108)
- [x] **Check the diff-ratio calculation against archived GPU evidence.** Rebuild the matched full and changed-row archives and reconcile the byte split and reconstruction receipts. [Offline smoke results](#smoke-check-on-archived-gpu-evidence)
- [ ] **Measure the post-Blend changed-row handoff.** Integrate the earlier final-row fingerprint, packing, and target verification flow. Check the byte split and fallback on the three-turn CLAPNQ smoke, then time both transfer arms with matched routing and retention. [GPU smoke](#first-evidencestore-gpu-smoke) · [Measurement design](#measure-the-speed-on-a-matched-replay)

Collect the follow-ups during the end-to-end review. After the review, implement the agreed changes together, verify the new events and byte counts in a small traced smoke run, then rerun the MuSiQue and MT-RAG workloads.

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
