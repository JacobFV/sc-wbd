#!/usr/bin/env bash
# Re-create the data symlinks a worktree needs.
#
# `assets` and `data` are symlinks into /data/scwbd and are deliberately
# git-ignored (ARCHITECTURE.md §1).  Because they are ignored and untracked,
# **merging master into a worktree deletes them** — git removes the paths that
# the untracking commits dropped, and nothing puts them back.  Every agent that
# merges master loses atlas and dataset access until they run this.
#
# It presents as a *missing dataset* rather than a broken link, so the next
# person to hit it misdiagnoses it.  That has now happened to three agents.
#
# Idempotent.  Run from anywhere; it links the worktree it lives in.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORE="/data/scwbd"

[ -d "$STORE" ] || { echo "error: $STORE not present on this machine" >&2; exit 1; }

for pair in "assets:$STORE/assets" "data:$STORE"; do
  name="${pair%%:*}"; target="${pair#*:}"
  link="$ROOT/$name"
  # A stale *directory* here is the failure mode that looks like missing data:
  # a test run can regenerate `assets` as a real 89 MB tree.  Only remove one
  # that is not already the symlink we want.
  if [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
    printf '  ok      %s -> %s\n' "$name" "$target"; continue
  fi
  if [ -e "$link" ] || [ -L "$link" ]; then
    printf '  replace %s (was: %s)\n' "$name" "$(readlink "$link" 2>/dev/null || echo 'real directory')"
    rm -rf "$link"
  else
    printf '  create  %s -> %s\n' "$name" "$target"
  fi
  ln -s "$target" "$link"
done

# Prove it resolves, rather than trusting that ln succeeded.
[ -e "$ROOT/assets/MANIFEST.json" ] || { echo "error: assets/MANIFEST.json unreachable after linking" >&2; exit 1; }
echo "verified: $ROOT/assets/MANIFEST.json resolves"
