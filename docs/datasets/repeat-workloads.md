# Repeat-heavy MuSiQue and MT-RAG workloads

Both selections use successful requests from the earlier 1P/3D native replay. A repeat is an occurrence of a versioned chunk that appeared in an earlier selected request. A shifted repeat has the same evidence tokens at a different prompt offset in an earlier selected request. The shifted counts below come from rerendering the archived retrieval results with the pinned Full Blend tokenizer and `¶` prompt format.

| Workload | Input | Evidence occurrences | Unique versioned chunks | Repeat occurrences | Shifted repeats |
| --- | --- | ---: | ---: | ---: | ---: |
| [MuSiQue single-turn 40](../../workloads/musique-single-turn-40.json) | 40 distinct questions | 240 | 31 | 209 (87.1%) | 208 |
| [MT-RAG multi-turn repeats](../../workloads/mtrag-multi-turn-repeat.json) | 6 complete CLAPNQ conversations, 40 turns | 240 | 132 | 108 (45.0%) | 91 |

The MuSiQue search compared 2,417 archived questions and kept the strongest 40-question set found. The replay order puts related requests near each other and retains all 40 unique question IDs. The MT-RAG search compared every eligible 40-turn subset of complete conversations within each archived domain.

| MT-RAG domain | Highest repeat count among eligible 40-turn subsets |
| --- | ---: |
| CLAPNQ | 108 |
| Govt | 94 |
| Cloud | 90 |
| FiQA | 66 |

CLAPNQ has **93 within-conversation** and **27 across-conversation** repeat occurrences. Twelve occurrences count in both. The legacy prompt preview includes 83 within-conversation and 20 across-conversation shifts. The six complete conversations have 7, 5, 9, 7, 6, and 6 turns. Each conversation runs in turn order with the dataset’s fixed reference history and retrieval rewrites.

The selection files record ordered request keys, hashes of the source queries and corpus, and expected evidence spans. The runner searches the full Faiss index for each selected question or turn. The report checks the default prompt hashes and evidence spans against the preview. Gated policies can change placement, so the report checks the retrieved and retained chunk identities and Faiss ranks. It verifies observed restores from the LMCache trace. LMCache availability also depends on the worker route, cache state, and the legacy connector’s lookup behavior.

## Dataset trace reviews

- [MuSiQue](musique.md) covers the full native and shared replays and the matched `default` and `seen-first` Full Blend runs.
- [MT-RAG](mtrag.md) covers the multi-turn CLAPNQ comparisons across `default`, `openclaw`, and `previous-turn-first`.

## Run the two workloads

From the [Full Blend runbook](../vllm-legacy-full-blend-runbook.md#start-from-the-login-node), set `workload=musique-single-turn-40` for the first run and `workload=mtrag-multi-turn-repeat` for the second. Both use the same 1B/3D layout and trace setting by default. The launcher appends two controlled probes after the 40 dataset requests. Those probes verify that the runtime can perform a shifted restore.
