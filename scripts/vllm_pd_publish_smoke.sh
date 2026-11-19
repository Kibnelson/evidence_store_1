#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
: "${RUN_ROOT:?Run this from the shell that ran the smoke}"
if [[ "$(git -C "$repo" branch --show-current)" != main ]]; then
  printf 'Publish from the shared checkout on main.\n' >&2
  exit 1
fi

settings=(--run-root "$RUN_ROOT" --topology "${SMOKE_TOPOLOGY:-1p3d}")
if [[ -n "${CHECK_ROOT:-}" ]]; then
  settings+=(--check-root "$CHECK_ROOT")
fi
archive="$("${PYTHON:-python3}" "$repo/scripts/vllm_pd_smoke_archive.py" "${settings[@]}")"
git -C "$repo" add -- "$archive"
if ! git -C "$repo" diff --cached --quiet -- "$archive"; then
  git -C "$repo" commit --only -m "Archive PD smoke ${archive##*/}" -- "$archive"
fi
git -C "$repo" fetch origin main
if ! git -C "$repo" diff --quiet FETCH_HEAD HEAD -- . ":(exclude)$archive"; then
  printf 'The local and remote branches differ outside %s. Review before pushing.\n' "$archive" >&2
  exit 1
fi
git -C "$repo" push origin main
printf 'Published %s at %s\n' "$archive" "$(git -C "$repo" rev-parse --short HEAD)"
