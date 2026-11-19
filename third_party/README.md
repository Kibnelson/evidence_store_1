# Pinned upstream sources

`sources.json` records each upstream URL, the branch or tag at the fork point, and the exact commit. The checkouts in `checkouts/` are shallow and detached with no Git remotes. The EvidenceStore repo's own remote is independent of these source checkouts.

| Checkout | Upstream ref | Fork-point commit |
| --- | --- | --- |
| [vLLM](https://github.com/vllm-project/vllm) | `main` | [`b24eda7127a9c5c317f699db3f580135eec7322b`](https://github.com/vllm-project/vllm/commit/b24eda7127a9c5c317f699db3f580135eec7322b) |
| [LMCache](https://github.com/LMCache/LMCache) | `dev` | [`042d47abdef0d460cf2cb0d5720e8a59280cda8a`](https://github.com/LMCache/LMCache/commit/042d47abdef0d460cf2cb0d5720e8a59280cda8a) |
| [Legacy LMCache](https://github.com/LMCache/LMCache_inprocess) | `main` | [`0aebefd026f5c479d3f6bcbee0d5facd1b2856ec`](https://github.com/LMCache/LMCache_inprocess/commit/0aebefd026f5c479d3f6bcbee0d5facd1b2856ec) |
| [SGLang](https://github.com/sgl-project/sglang) | `v0.5.23` | [`297e54fd105e974cdf401e5b4d20253a22c16f90`](https://github.com/sgl-project/sglang/commit/297e54fd105e974cdf401e5b4d20253a22c16f90) |
| [Mooncake](https://github.com/kvcache-ai/Mooncake) | `v0.3.14` | [`e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c`](https://github.com/kvcache-ai/Mooncake/commit/e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c) |
| [PPD](https://github.com/Stark-X6/vllm-ppd) | `main` | [`28aaa63c6d7a0a0e00d97f8c958291bb6b6a4367`](https://github.com/Stark-X6/vllm-ppd/commit/28aaa63c6d7a0a0e00d97f8c958291bb6b6a4367) |
| [NIXL](https://github.com/ai-dynamo/nixl) | `main` | [`e4d435d072d503eaee1baf8450b8f4d33c5c6e9f`](https://github.com/ai-dynamo/nixl/commit/e4d435d072d503eaee1baf8450b8f4d33c5c6e9f) |
| [Faiss](https://github.com/facebookresearch/faiss) | `main` | [`14392c952788ca1d6980361cf01d1c8736afc4db`](https://github.com/facebookresearch/faiss/commit/14392c952788ca1d6980361cf01d1c8736afc4db) |
| [MuSiQue](https://github.com/StonyBrookNLP/musique) | `main` | [`922ac98f19a201998dbdae6d7f2887a5258dbdeb`](https://github.com/StonyBrookNLP/musique/commit/922ac98f19a201998dbdae6d7f2887a5258dbdeb) |
| [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) | `main` | [`13800e5be57df1b4040b9b1588c6c811779e69e9`](https://github.com/Alab-NII/2wikimultihop/commit/13800e5be57df1b4040b9b1588c6c811779e69e9) |
| [IBM MT-RAG](https://github.com/IBM/mt-rag-benchmark) | `main` | [`e0970de1b1207c2a9506adf238e6e1a22037e38a`](https://github.com/IBM/mt-rag-benchmark/commit/e0970de1b1207c2a9506adf238e6e1a22037e38a) |

From the repo root:

```sh
python3 scripts/checkout_sources.py
python3 scripts/checkout_sources.py --verify
```

These are source-inspection pins. The checkout command leaves Git LFS payloads and submodules for the build setup. The runtime tuple also needs compatible packages, model and dataset assets, image digests, CUDA environment, and the private multi-process CacheBlend plugin. Record those in the run manifest as each setup passes GPU bring-up. PPD's bundled `vllm-source` tree is `479990207360b9ea055ffe864fbd8352816979af` at the pinned PPD commit. Retain that tree identity independently of the plain vLLM checkout.

## Editable vLLM fork

The standalone `third_party/forks/vllm/` checkout has a local `evidencestore/native-pd` branch and zero remotes. Edit and commit changes there. The [fork receipt](vllm-fork.json) identifies the upstream fork point and the full source commit exported for the allocation. The Git bundle transports that committed history to a fresh checkout.

```sh
python3 scripts/checkout_sources.py vllm
python3 scripts/vllm_pd_prepare.py

# Refresh the portable bundle after committing source changes in the fork.
python3 scripts/vllm_pd_prepare.py --export
```

Carry `vllm-native-pd.bundle` and `vllm-fork.json` with the experiment repo. The prepare command can fast-forward a clean local branch to a newer bundled commit. The launcher records the current local commit, tree, and any working-tree changes to identify the source measured. [Allocation runbook](../docs/vllm-native-pd-runbook.md)

## Editable LMCache fork

`scripts/lmcache_pd_prepare.py` restores the no-remote LMCache fork from `lmcache-pd.bundle` and verifies its [source receipt](lmcache-pd-fork.json). The source gates `EVIDENCESTORE_TRACE_LMCACHE_TRANSFERS=1` and `EVIDENCESTORE_TRACE_LMCACHE_BLEND=1` enable ordinary transfer and Blend lookup and scatter events, respectively. The Blend smoke sets both gates with its `--trace` option. [Blend smoke runbook](../docs/vllm-native-pd-runbook.md#switch-prefix-prefill-to-blend-on-one-lmcache-server)
