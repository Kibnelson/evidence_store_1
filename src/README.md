# Project code

`evidencestore` holds the native gateway, Faiss retrieval, trace writing, and request-level reporting. The worker and NIXL hooks live in the editable vLLM fork at `third_party/forks/vllm/`. `EVIDENCESTORE_PD_TRACE=1` enables the source hooks. The event contract carries request IDs, chunk evidence, token spans, and transfer payload through the before-and-after comparisons. [vLLM runbook](../docs/vllm-native-pd-runbook.md)
