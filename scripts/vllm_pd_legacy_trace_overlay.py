#!/usr/bin/env python3
"""Add gated measurement hooks to the allocation-local Full Blend build copies."""

import argparse
import hashlib
import json
from pathlib import Path


VLLM_HOOK = "from vllm._evidencestore_legacy_trace import ENABLED as _legacy_trace_on\n"
LM_CACHE_HOOK = "from vllm._evidencestore_legacy_trace import ENABLED as _es_legacy_trace_on\n"
NIXL = "vllm/distributed/kv_transfer/kv_connector/v1/nixl"


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def replace_once(text, old, new, path):
    if text.count(old) != 1:
        raise ValueError(f"Historical trace anchor must occur once in {path}: {old!r}")
    return text.replace(old, new, 1)


def _plan(root, name, anchor, imports, edits):
    path = root / name
    original = path.read_text(encoding="utf-8")
    if "_evidencestore_legacy_trace" in original:
        raise ValueError(f"Use a fresh build copy for the legacy overlay: {path}")
    updated = replace_once(original, anchor, imports + anchor, path)
    for old, new in edits:
        updated = replace_once(updated, old, new, path)
    compile(updated, str(path), "exec")
    return path, updated, {"path": name, "before_sha256": _sha(original), "after_sha256": _sha(updated)}


def _edit_vllm(root):
    anchor = "from vllm.logger import init_logger\n"
    emit = VLLM_HOOK + "from vllm._evidencestore_legacy_trace import emit as _legacy_emit\n"
    return [
        _plan(root, "vllm/v1/engine/input_processor.py", anchor, emit, [(
            '            request.request_id = f"{request.external_req_id}-{random_uuid():.8}"\n',
            '            request.request_id = f"{request.external_req_id}-{random_uuid():.8}"\n'
            '        if _legacy_trace_on:\n'
            '            _legacy_emit("engine_request_id", native_request_id=request.request_id,\n'
            '                         external_request_id=request.external_req_id)\n',
        )]),
        _plan(root, "vllm/v1/core/sched/scheduler.py", anchor, emit, [(
            "            request.num_computed_tokens += num_scheduled_token\n",
            "            if _legacy_trace_on:\n"
            "                _legacy_start = request.num_computed_tokens\n"
            '                _legacy_emit("engine_token_work", native_request_id=req_id,\n'
            "                             position_start=_legacy_start,\n"
            "                             position_end=_legacy_start + num_scheduled_token,\n"
            "                             prompt_tokens=request.num_prompt_tokens)\n"
            "            request.num_computed_tokens += num_scheduled_token\n",
        )]),
        _plan(root, "vllm/v1/worker/gpu_model_runner.py", anchor,
              VLLM_HOOK + "from vllm._evidencestore_legacy_trace import gpu_forward_end, gpu_forward_start\n", [(
            "            model_output = self._model_forward(\n",
            "            _legacy_gpu_ticket = gpu_forward_start(torch, scheduler_output) if _legacy_trace_on else None\n"
            "            model_output = self._model_forward(\n",
        ), (
            "                **model_kwargs,\n            )\n\n"
            '        with record_function_or_nullcontext("gpu_model_runner: postprocess"):\n',
            "                **model_kwargs,\n            )\n"
            "            if _legacy_gpu_ticket is not None:\n"
            "                gpu_forward_end(_legacy_gpu_ticket)\n\n"
            '        with record_function_or_nullcontext("gpu_model_runner: postprocess"):\n',
        )]),
        _plan(root, f"{NIXL}/scheduler.py", anchor, emit, [(
            "            remote_num_tokens = request.num_computed_tokens\n",
            "            remote_num_tokens = request.num_computed_tokens\n"
            "            if _legacy_trace_on and is_p_node:\n"
            '                _legacy_emit("engine_kv_handoff_ready", native_request_id=request.request_id,\n'
            "                             remote_token_count=remote_num_tokens,\n"
            "                             remote_block_ids=block_ids)\n",
        )]),
        _plan(root, f"{NIXL}/worker.py", anchor,
              VLLM_HOOK + "from vllm._evidencestore_legacy_trace import read_done, read_failed, read_submitted\n", [(
            "            self._recving_transfers[request_id].append(handle)\n",
            "            self._recving_transfers[request_id].append(handle)\n"
            "            if _legacy_trace_on:\n"
            "                read_submitted(self, handle, request_id, remote_request_id, local_block_ids,\n"
            "                               remote_block_ids, remote_info.remote_block_size, remote_physical_per_logical)\n",
        ), (
            "                        self.xfer_stats.record_transfer(res)\n",
            "                        self.xfer_stats.record_transfer(res)\n"
            "                        if _legacy_trace_on:\n"
            "                            read_done(self, handle, res)\n",
        ), (
            "        self.xfer_stats.record_failed_transfer()\n",
            "        self.xfer_stats.record_failed_transfer()\n"
            "        if _legacy_trace_on:\n"
            "            read_failed(self, handle, req_id)\n",
        )]),
    ]


def _edit_lmcache(root):
    anchor = "from lmcache.logging import init_logger\n"
    emit = LM_CACHE_HOOK + "from vllm._evidencestore_legacy_trace import emit as _es_legacy_emit\n"
    storage_hooks = (
        "from vllm._evidencestore_legacy_storage_trace import (\n"
        "    ENABLED as _es_storage_trace_on,\n"
        "    sample_put as _es_sample_storage_put,\n"
        "    sample_restore as _es_sample_storage_restore,\n"
        ")\n"
    )
    page_hooks = (
        "from vllm._evidencestore_legacy_page_trace import (\n"
        "    ENABLED as _es_page_trace_on,\n"
        "    sample_put as _es_sample_page_put,\n"
        "    start_restore as _es_start_page_restore,\n"
        ")\n"
    )
    return [
        _plan(root, "lmcache/v1/cache_engine.py", anchor, emit + storage_hooks + page_hooks, [(
            "            t_start = time.perf_counter()\n            mem_obj_generator = self.gpu_connector.batched_from_gpu(\n",
            "            t_start = time.perf_counter()\n"
            "            if _es_legacy_trace_on:\n"
            "                _es_store_started = time.monotonic_ns()\n"
            "                _es_store_ranges = [\n"
            '                    {"start_token": start, "end_token": end,\n'
            '                     "bytes": sum(layer[index].get_size() for layer in memory_objs)}\n'
            "                    for index, (start, end) in enumerate(zip(starts, ends, strict=True))\n"
            "                ]\n"
            "            mem_obj_generator = self.gpu_connector.batched_from_gpu(\n",
        ), (
            "                    keys[layer_id], memory_objs[layer_id], location=self.store_location\n                )\n",
            "                    keys[layer_id], memory_objs[layer_id], location=self.store_location\n                )\n"
            "                if _es_storage_trace_on and layer_id in (0, self.num_layers // 2, self.num_layers - 1):\n"
            "                    _es_sample_storage_put(self.storage_manager, self.store_location, req_id, layer_id,\n"
            "                                           keys[layer_id], memory_objs[layer_id], starts, ends, tokens)\n"
            "            if _es_page_trace_on:\n"
            "                _es_sample_page_put(self, req_id, keys, memory_objs, starts, ends, tokens, kwargs[\"slot_mapping\"])\n"
            "\n            if _es_legacy_trace_on:\n"
            '                _es_legacy_emit("legacy_lmcache_d2h_complete", component="lmcache",\n'
            "                                native_request_id=req_id, started_ns=_es_store_started,\n"
            "                                mapped_ranges=_es_store_ranges, layers=self.num_layers)\n",
        ), (
            "            tier_sources: list[dict[str, Any]] = []\n",
            "            tier_sources: list[dict[str, Any]] = []\n"
            "            if _es_legacy_trace_on:\n"
            "                _es_load_started = time.monotonic_ns()\n"
            "                _es_load_bytes = [0] * len(starts)\n"
            "                _es_cached_positions = []\n"
            "            if _es_page_trace_on:\n"
            "                _es_page_restore = None\n",
        ), (
            "                    mem_objs_layer = task.result()\n",
            "                    mem_objs_layer = task.result()\n"
            "                    if _es_legacy_trace_on:\n"
            "                        for _es_index, _es_obj in enumerate(mem_objs_layer):\n"
            "                            _es_load_bytes[_es_index] += _es_obj.get_size()\n"
            "                            if layer_id == 0:\n"
            "                                _es_cached_positions.append(_es_obj.metadata.cached_positions)\n"
            "                    if _es_storage_trace_on and layer_id in (0, self.num_layers // 2, self.num_layers - 1):\n"
            "                        _es_sample_storage_restore(self.storage_manager, location, req_id, layer_id,\n"
            "                                                   keys_layer_major[layer_id], mem_objs_layer, starts, ends)\n"
            "                    if _es_page_trace_on:\n"
            "                        if layer_id == 0:\n"
            "                            _es_page_restore = _es_start_page_restore(\n"
            "                                self, req_id, location, keys_layer_major, starts, ends, kwargs[\"slot_mapping\"]\n"
            "                            )\n"
            "                        if _es_page_restore is not None:\n"
            "                            _es_page_restore.sample_layer(layer_id, mem_objs_layer)\n",
        ), (
            "                _consumer_terminal_complete = True\n",
            "                _consumer_terminal_complete = True\n"
            "                if _es_page_trace_on and _es_page_restore is not None:\n"
            "                    _es_page_restore.finish()\n"
            "                if _es_legacy_trace_on:\n"
            "                    _es_load_ranges = []\n"
            "                    for _es_start, _es_end, _es_bytes, _es_old in zip(\n"
            "                        starts, ends, _es_load_bytes, _es_cached_positions, strict=True,\n"
            "                    ):\n"
            "                        _es_old = _es_old.tolist() if _es_old is not None else []\n"
            "                        _es_old_start = _es_old[0] if _es_old else None\n"
            "                        if (_es_old and _es_old != list(range(\n"
            "                                _es_old_start, _es_old_start + _es_end - _es_start))):\n"
            "                            _es_old_start = None\n"
            "                        _es_load_ranges.append({\n"
            '                            "start_token": _es_start, "end_token": _es_end,\n'
            '                            "bytes": _es_bytes, "old_start_token": _es_old_start,\n'
            "                        })\n"
            '                    _es_legacy_emit("legacy_lmcache_h2d_complete", component="lmcache",\n'
            "                                    native_request_id=req_id, started_ns=_es_load_started,\n"
            "                                    mapped_ranges=_es_load_ranges, layers=self.num_layers,\n"
            "                                    backend=location)\n",
        )]),
        _plan(root, "lmcache/v1/compute/blend/blender.py", anchor,
              LM_CACHE_HOOK + "from vllm._evidencestore_legacy_trace import blend_complete as _es_blend_complete\n", [(
            "        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)\n",
            "        _es_layer_positions = [] if _es_legacy_trace_on else None\n"
            "        if _es_layer_positions is not None:\n"
            "            _es_started = time.monotonic_ns()\n"
            "        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)\n",
        ), (
            "                next(layerwise_model_executor)\n",
            "                next(layerwise_model_executor)\n"
            "                if _es_layer_positions is not None:\n"
            "                    _es_layer_positions.append(self.metadata.positions)\n",
        ), (
            "            retrieval_complete = True\n\n            self.metadata.clean()\n",
            "            retrieval_complete = True\n\n"
            "            if _es_layer_positions is not None:\n"
            '                _es_blend_complete(kwargs["req_id"], tokens, _es_layer_positions, _es_started, torch)\n'
            "            self.metadata.clean()\n",
        )]),
    ]


def install(vllm_source, lmcache_source):
    planned = {"vllm": _edit_vllm(vllm_source), "lmcache": _edit_lmcache(lmcache_source)}
    for source_name, module_name in (
        ("vllm_pd_legacy_trace_events.py", "vllm/_evidencestore_legacy_trace.py"),
        ("vllm_pd_legacy_storage_trace.py", "vllm/_evidencestore_legacy_storage_trace.py"),
        ("vllm_pd_legacy_page_trace.py", "vllm/_evidencestore_legacy_page_trace.py"),
    ):
        module = vllm_source / module_name
        helper = Path(__file__).with_name(source_name).read_text(encoding="utf-8")
        compile(helper, str(module), "exec")
        if module.exists():
            raise ValueError(f"Legacy trace helper already exists: {module}")
        planned["vllm"].append((module, helper, {"path": module_name, "after_sha256": _sha(helper)}))
    for paths in planned.values():
        for path, text, _ in paths:
            path.write_text(text, encoding="utf-8")
    return {name: [receipt for _, _, receipt in paths] for name, paths in planned.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--lmcache-source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.vllm_source, args.lmcache_source), indent=2))


if __name__ == "__main__":
    main()
