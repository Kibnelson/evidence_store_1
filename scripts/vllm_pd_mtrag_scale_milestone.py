#!/usr/bin/env python3
"""Publish an aggregate CPU replay report for a verified MT-RAG campaign prefix."""

import argparse
from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import re
from statistics import median
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit

from vllm_pd_mtrag_scale_git import PART_BYTES
from vllm_pd_mtrag_scale_publish import (
    GitCheckpoints, PART_BYTES as RELEASE_PART_BYTES, RELEASE_REPO, SCHEMA, canonical, get_checkpoint, validate_plan,
)


POLICIES = ("independent", "forward_pinned")
SUMMED = ("hits", "misses", "delta_hits", "target_reads_bytes", "cold_base_reads_bytes",
          "rewrites", "rewritten_bytes", "rejected_admissions")
TOTAL_FIELDS = ("turns", "stores", "observed_loads", "unique_raw_pages", "raw_capture_bytes",
                "archive_bytes", "archive_parts", "standalone_frame_bytes", "eligible_exponent_targets")
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_ENTRY = re.compile(rb"100644 blob ([0-9a-f]{40}|[0-9a-f]{64})\t(.+)")


def nonnegative(value):
    return type(value) is int and value >= 0


def tree_blobs(checkpoints):
    entries = checkpoints.git("ls-tree", "-rz", "FETCH_HEAD", "--", checkpoints.prefix).stdout
    blobs = {}
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        match = GIT_ENTRY.fullmatch(entry)
        if not match:
            raise ValueError(f"Unexpected Git tree entry under {checkpoints.prefix}")
        blobs[match[2].decode()] = match[1].decode()
    return blobs


def fetch_studies(checkpoints, paths, blobs):
    missing = [path for path in paths if path not in blobs]
    if missing:
        raise ValueError(f"Milestone is missing {len(missing)} capture or study files, starting with {missing[0]}")
    # Match Git's promisor fetch, requesting only the exact small metadata objects.
    objects = "\n".join(sorted({blobs[path] for path in paths})) + "\n"
    checkpoints.git("-c", "fetch.negotiationAlgorithm=noop", "fetch", "origin", "--quiet", "--no-tags",
                    "--no-write-fetch-head", "--recurse-submodules=no", "--filter=blob:none", "--stdin",
                    data=objects.encode())


def validate_study(study, capture):
    conversation = capture["conversation_id"]
    config = study["input"]["configuration"]
    capacities = config["capacities_mib"]
    if (not capacities or any(not nonnegative(n) or not n for n in capacities)
            or capacities != sorted(set(capacities)) or not HEX_SHA256.fullmatch(config["implementation_sha256"])):
        raise ValueError(f"Invalid CPU replay configuration for {conversation}")
    scope_fields = (("natural_requests", "turns"), ("captured_stores", "stores"), ("native_loads", "observed_loads"))
    if any(study["scope"][metric] != capture[field] for metric, field in scope_fields):
        raise ValueError(f"CPU replay scope differs from capture for {conversation}")
    results = study["capacity_results"]
    tested = sorted(int(capacity) for capacity in results)
    if (any(n <= 0 or str(n) not in results for n in tested) or not set(capacities).issubset(tested)
            or any(not nonnegative(value) for value in (study["standalone_frame_bytes"], study["eligible_exponent_targets"]))):
        raise ValueError(f"Invalid CPU replay results for {conversation}")
    for capacity in tested:
        for policy in POLICIES:
            row = results[str(capacity)][policy]
            if (any(not nonnegative(row[field]) for field in (*SUMMED, "verified_hits"))
                    or row["hits"] + row["misses"] != capture["observed_loads"]
                    or max(row["delta_hits"], row["verified_hits"]) > row["hits"]):
                raise ValueError(f"CPU replay load accounting differs for {conversation}, {policy}, {capacity} MiB")
            if policy == "independent" and any(row[field] for field in (
                "delta_hits", "cold_base_reads_bytes", "rewrites", "rewritten_bytes"
            )):
                raise ValueError(f"Standalone replay reports differential work for {conversation}, {capacity} MiB")
    audit = {}
    for policy in POLICIES:
        first = None
        if capture["observed_loads"]:
            first = next((n for n in tested if results[str(n)][policy]["misses"] == 0), None)
            budget = first if first is not None else capacities[-1]
            verified = results[str(budget)][policy]
            if verified["hits"] != verified["verified_hits"]:
                raise ValueError(f"Expected exact-byte replay at the verification budget for {conversation}, {policy}")
            audit[policy] = {"capacity_mib": budget, "verified_hits": verified["verified_hits"]}
        if study["first_full_capacity_mib"][policy] != first:
            raise ValueError(f"First full-hit capacity is inconsistent for {conversation}, {policy}")
    return audit


def summarize(items, capacities):
    totals = {field: sum(item[field] for item in items) for field in TOTAL_FIELDS}
    totals["conversations"] = len(items)
    totals["zero_load_conversations"] = sum(item["observed_loads"] == 0 for item in items)
    capacity_results = {
        str(n): {policy: {field: sum(item["results"][str(n)][policy][field] for item in items) for field in SUMMED}
                 for policy in POLICIES} for n in capacities
    }
    measured = [item for item in items if item["observed_loads"]]
    first_full = {}
    for policy in POLICIES:
        budgets = [item["first_full_capacity_mib"][policy] for item in measured]
        first_full[policy] = {
            "conversations_by_capacity_mib": {
                str(n): count for n, count in sorted(Counter(n for n in budgets if n is not None).items())
            },
            "conversations_with_loads_and_no_full_hit": budgets.count(None),
            "zero_load_conversations": totals["zero_load_conversations"],
            "verified_hits_at_audit_budgets": sum(item["exactness_audit"][policy]["verified_hits"] for item in measured),
        }
    differences = []
    for item in measured:
        independent, pinned = (item["first_full_capacity_mib"][name] for name in POLICIES)
        if independent is not None and pinned is not None:
            differences.append(pinned - independent)
    first_full["paired_forward_pinned_minus_independent"] = {
        "both_full": len(differences),
        "with_loads_without_both_full": len(measured) - len(differences),
        "forward_pinned_lower": sum(n < 0 for n in differences),
        "equal": differences.count(0),
        "forward_pinned_higher": sum(n > 0 for n in differences),
        "min_mib": min(differences, default=None),
        "p50_mib": median(differences) if differences else None,
        "max_mib": max(differences, default=None),
    }
    return {"totals": totals, "capacity_results": capacity_results, "first_full": first_full}


def summarize_timings(items):
    groups = defaultdict(list)
    for item in items:
        config = item["config"]
        key = (config["host"], config["platform"], config["python"], str(config["zstd"]))
        groups[key].extend(item["timing_pairs"])
    result = []
    for (host, system, python, zstd), pairs in sorted(groups.items()):
        values = defaultdict(list)
        for pair in pairs:
            for operation, time_ns in pair["median_cpu_ns"].items():
                if not isinstance(time_ns, (int, float)) or not math.isfinite(time_ns) or time_ns < 0:
                    raise ValueError(f"Invalid replay timing for {host}, {operation}")
                values[operation].append(time_ns)
        result.append({"host": host, "platform": system, "python": python, "zstd": zstd,
                       "matched_pairs": len(pairs), "operations": {
                           operation: {"pairs": len(times), "p50_ms": median(times) / 1e6,
                                       "p95_ms": sorted(times)[math.ceil(0.95 * len(times)) - 1] / 1e6}
                           for operation, times in values.items()
                       }})
    return result


def archive_inputs(plan, archive, blobs):
    if plan.get("archive_backend") == "git-ssh":
        inputs = {}
        for part in archive["parts"]:
            path = part["git_path"]
            if (part["id"] != path or not nonnegative(part["bytes"]) or not 0 < part["bytes"] <= PART_BYTES
                    or not HEX_SHA256.fullmatch(part["sha256"]) or blobs.get(path) != part["git_blob"]):
                raise ValueError(f"Remote Git payload membership differs at {path}")
            inputs[path] = part["git_blob"]
        return inputs
    for part in archive["parts"]:
        link = urlsplit(part["url"])
        expected_path = f"/{plan['release_repo']}/releases/download/{plan['release_tag']}/{part['name']}"
        if (not nonnegative(part["id"]) or not part["id"] or not nonnegative(part["bytes"])
                or not 0 < part["bytes"] <= RELEASE_PART_BYTES or not HEX_SHA256.fullmatch(part["sha256"])
                or not re.fullmatch(r"[a-zA-Z0-9_.-]+", part["name"]) or link.scheme != "https"
                or link.netloc != "github.com" or unquote(link.path) != expected_path or link.query or link.fragment):
            raise ValueError(f"Published Release payload receipt differs for {part['name']}")
    return {}


def collect(checkpoints, plan, selected, blobs):
    paths = [(checkpoints.capture_path(row["conversation_id"]), checkpoints.study_path(row["conversation_id"]))
             for row in selected]
    fetch_studies(checkpoints, [path for pair in paths for path in pair], blobs)
    inputs, items = {}, []
    for row, (capture_path, study_path) in zip(selected, paths, strict=True):
        conversation = row["conversation_id"]
        capture = get_checkpoint(checkpoints, None, plan, conversation, metadata_only=True)
        if capture is None or not HEX_SHA256.fullmatch(capture["study_sha256"] or ""):
            raise ValueError(f"No verified capture and CPU study for {conversation}")
        if any(not nonnegative(capture[field]) for field in (
            "turns", "stores", "observed_loads", "unique_raw_pages", "raw_capture_bytes"
        )):
            raise ValueError(f"Invalid capture counts for {conversation}")
        archive = capture["archive"]
        inputs.update({capture_path: blobs[capture_path], study_path: blobs[study_path]})
        inputs.update(archive_inputs(plan, archive, blobs))
        study = checkpoints.read(study_path)
        audit = validate_study(study, capture)
        items.append({**{field: capture[field] for field in (
            "conversation_id", "domain", "turn_window", "turns", "stores", "observed_loads",
            "unique_raw_pages", "raw_capture_bytes", "study_sha256",
        )}, "capture_path": capture_path, "capture_git_blob": blobs[capture_path],
            "study_path": study_path, "study_git_blob": blobs[study_path], "archive_sha256": archive["sha256"],
            "archive_bytes": archive["bytes"], "archive_parts": len(archive["parts"]),
            "config": study["input"]["configuration"], "results": study["capacity_results"],
            "timing_pairs": study["matched_pairs"], "first_full_capacity_mib": study["first_full_capacity_mib"],
            "exactness_audit": audit, "standalone_frame_bytes": study["standalone_frame_bytes"],
            "eligible_exponent_targets": study["eligible_exponent_targets"]})
    return items, inputs


def make_report(checkpoints, plan, through, items):
    configurations = [item["config"] for item in items]
    common = sorted(set.intersection(*(set(config["capacities_mib"]) for config in configurations)))
    settings = ("schema_version", "implementation_sha256", "minimum_saving_pct")
    basis = {name: configurations[0][name] for name in settings}
    if not common or any({name: config[name] for name in settings} != basis for config in configurations):
        raise ValueError("Captured CPU studies need a compatible replay implementation and common capacity budget")
    domains = defaultdict(list)
    for item in items:
        domains[item["domain"]].append(item)
    return {"schema_version": 1, "campaign": plan["campaign"], "source_commit": plan["source_commit"],
            "plan_sha256": plan["plan_sha256"], "plan_git_path": checkpoints.plan_path,
            "results_branch": checkpoints.branch.removeprefix("refs/heads/"),
            "archive_backend": plan.get("archive_backend", "release"),
            "through": through, "common_capacities_mib": common, "replay_basis": basis,
            **summarize(items, common),
            "domains": {domain: summarize(rows, common) for domain, rows in domains.items()},
            "matched_cpu_timings_by_host": summarize_timings(items),
            "captures": [{key: value for key, value in item.items() if key not in {"config", "results", "timing_pairs"}}
                         for item in items]}


def run(args, scratch):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", args.campaign):
        raise ValueError("Campaign IDs must contain lowercase ASCII letters, digits, and hyphens")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", args.repo):
        raise ValueError("Expected an owner/repository for the Git destination")
    ssh = f"git@github.com:{args.repo}.git"
    checkpoints = GitCheckpoints(scratch, args.repo_root, args.campaign,
                                 args.git_fetch_url or ssh, args.git_push_url or ssh)
    checkpoints.refresh()
    plan = checkpoints.read(checkpoints.plan_path)
    if (not plan or plan["schema_version"] != SCHEMA or plan["campaign"] != args.campaign
            or plan["plan_sha256"] != hashlib.sha256(canonical(plan["plan"])).hexdigest()):
        raise ValueError("Remote plan does not match this campaign")
    backend = plan.get("archive_backend", "release")
    if backend == "git-ssh":
        expected = plan["archive_repo"] == args.repo
    else:
        expected = (backend == "release" and plan["release_repo"] == args.repo
                    and plan["release_tag"] == args.campaign and nonnegative(plan["release_id"]) and plan["release_id"])
    if not expected:
        raise ValueError("Remote plan does not match the campaign archive destination")
    validate_plan(plan["plan"])
    queue = plan["plan"]["conversations"]
    if not 1 <= args.through <= len(queue):
        raise ValueError(f"--through must be between 1 and {len(queue)}")
    blobs = tree_blobs(checkpoints)
    items, inputs = collect(checkpoints, plan, queue[:args.through], blobs)
    inputs[checkpoints.plan_path] = blobs[checkpoints.plan_path]
    report = make_report(checkpoints, plan, args.through, items)
    target = f"{checkpoints.prefix}/milestones/{args.through:03d}.json"
    checkpoints.refresh()
    fresh = tree_blobs(checkpoints)
    if any(fresh.get(path) != oid for path, oid in inputs.items()):
        raise ValueError("Remote milestone inputs changed during verification. Retry with the current results branch.")
    previous = checkpoints.read(target)
    if previous is None:
        checkpoints.write({target: report}, f"Report MT-RAG milestone {args.campaign} through {args.through}")
    elif previous != report:
        raise ValueError(f"Remote milestone differs at {target}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--through", required=True, type=int)
    parser.add_argument("--repo", default=RELEASE_REPO, help="GitHub owner/repository")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--git-fetch-url", help="Override SSH fetch URL")
    parser.add_argument("--git-push-url", help="Override SSH push URL")
    args = parser.parse_args()
    try:
        with TemporaryDirectory(prefix="mtrag-milestone-", dir=args.scratch_dir) as work:
            report = run(args, Path(work).resolve())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"MT-RAG milestone stopped. {error}\n")
    print(canonical(report).decode(), end="")


if __name__ == "__main__":
    main()
