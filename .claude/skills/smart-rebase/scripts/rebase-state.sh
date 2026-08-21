#!/usr/bin/env bash
# rebase-state.sh
#
# Read-only helper for the smart-rebase skill. Reports whether a rebase is
# currently in progress in this working tree, what it's rebasing onto, and
# which paths git currently considers conflicted (unmerged).
#
# Takes NO arguments, by design -- with nothing to parse, there is no
# argument-shaped surface for a flag- or path-injection exploit to attach
# to. Every command below is read-only plumbing (rev-parse, status, cat on
# git's own internal state files); nothing here writes to the working tree,
# the index, or history.
#
# Usage: bash rebase-state.sh

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

GIT_DIR="$(git rev-parse --git-dir)"

echo "== Current branch =="
git rev-parse --abbrev-ref HEAD
echo

STATE_DIR=""
if [[ -d "$GIT_DIR/rebase-merge" ]]; then
  STATE_DIR="$GIT_DIR/rebase-merge"
elif [[ -d "$GIT_DIR/rebase-apply" ]]; then
  STATE_DIR="$GIT_DIR/rebase-apply"
fi

if [[ -z "$STATE_DIR" ]]; then
  echo "== No rebase in progress =="
  echo "(nothing under $GIT_DIR/rebase-merge or $GIT_DIR/rebase-apply)"
else
  echo "== Rebase in progress (state dir: $STATE_DIR) =="
  if [[ -f "$STATE_DIR/onto" ]]; then
    ONTO_HASH="$(cat "$STATE_DIR/onto")"
    echo "Rebasing onto commit: $ONTO_HASH"
    git show -s --format='  %h  %ad  %s' --date=short "$ONTO_HASH" 2>/dev/null || true
  fi
  if [[ -f "$STATE_DIR/head-name" ]]; then
    echo "Original branch being replayed: $(cat "$STATE_DIR/head-name")"
  fi
  if [[ -f "$STATE_DIR/msgnum" && -f "$STATE_DIR/end" ]]; then
    echo "Step: $(cat "$STATE_DIR/msgnum") of $(cat "$STATE_DIR/end")"
  fi
fi
echo

echo "== Unmerged (conflicted) paths, per 'git status' =="
UNMERGED="$(git status --porcelain=v1 | awk '$1 ~ /^(DD|AU|UD|UA|DU|AA|UU)$/ {print}' || true)"
if [[ -z "$UNMERGED" ]]; then
  echo "(none)"
else
  echo "$UNMERGED"
fi
echo
echo "For any path listed above, run show-conflict.sh on it to see both sides of the conflict."
