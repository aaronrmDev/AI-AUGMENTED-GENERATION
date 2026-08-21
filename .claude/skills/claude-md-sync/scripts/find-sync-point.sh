#!/usr/bin/env bash
# find-sync-point.sh
#
# Read-only helper for the claude-md-sync skill. It performs the mechanical
# first half of the AUTOLEARNING.md contract: locate the last commit that
# touched CLAUDE.md or anything under docs/ (the last time a human, or an
# agent acting on an already-approved proposal, actually reconciled the docs
# with reality), then list everything that has happened since.
#
# This script NEVER mutates repo or working-tree state. It runs nothing but
# read-only git plumbing (log, show, diff --stat, ls-files). If you are
# extending this script, keep it that way — do not add `git add`, `git
# commit`, `git push`, or anything else that writes.
#
# Usage: bash find-sync-point.sh   (run from anywhere inside the repo)

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BASELINE="$(git log -1 --format=%H -- CLAUDE.md docs/ 2>/dev/null || true)"

if [ -z "$BASELINE" ]; then
  echo "== No sync point found =="
  echo "No commit in this repo's history has ever touched CLAUDE.md or docs/."
  echo "Treating the entire project history as unaccounted for."
  echo
  echo "== Full commit history =="
  git log --format='%h  %ad  %s' --date=short
  echo
  echo "== Current docs/ tree (tracked files) =="
  git ls-files docs/ CLAUDE.md
  exit 0
fi

echo "== Sync point =="
echo "Last commit to touch CLAUDE.md or docs/:"
git show -s --format='  %h  %ad  %s' --date=short "$BASELINE"
echo

echo "== Commits since the sync point (oldest first) =="
COMMITS_SINCE="$(git log --format='%h  %ad  %s' --date=short --reverse "${BASELINE}..HEAD")"
if [ -z "$COMMITS_SINCE" ]; then
  echo "(none — HEAD is the sync-point commit itself; nothing has happened since the last doc sync)"
else
  echo "$COMMITS_SINCE"
fi
echo

echo "== Files changed since the sync point, with stats =="
DIFFSTAT="$(git diff --stat "${BASELINE}..HEAD" 2>/dev/null || true)"
if [ -z "$DIFFSTAT" ]; then
  echo "(no changes)"
else
  echo "$DIFFSTAT"
fi
echo

echo "== Current docs/ tree (tracked files) =="
git ls-files docs/ CLAUDE.md
