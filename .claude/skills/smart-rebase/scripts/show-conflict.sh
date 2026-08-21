#!/usr/bin/env bash
# show-conflict.sh <path>
#
# Read-only helper for the smart-rebase skill. Shows the three sides of one
# conflicting path during an in-progress rebase, so both sides of a hunk can
# be read and reasoned about before proposing a resolution.
#
# WHY THIS SCRIPT VALIDATES SO STRICTLY (same reasoning as claude-md-sync's
# show-commit.sh): 'git show :N:<path>' can be made to read any blob in the
# repository, and a careless caller could be tricked into passing something
# that isn't really "the other side of an active conflict" -- a path with a
# leading '-' that git would parse as a flag, a '..' segment attempting to
# escape the repo-relative namespace, or simply a path nobody is actually
# conflicted on right now. This script closes all three:
#
#   1. Shape check -- the path must match a plain repo-relative pattern with
#      no leading '-' and no '..' segment, before it ever reaches git.
#   2. Defense in depth -- even a shape-valid path is refused unless 'git
#      status' *right now* reports that exact path as unmerged. That keeps
#      this script's real capability limited to "inspect an actual,
#      currently-open conflict," not "dump any blob in the repository."
#
# Neither check is optional, and neither should be loosened to make an edge
# case more convenient -- that would defeat the reason this script exists.
#
# IMPORTANT GIT GOTCHA -- read before interpreting the output: during a
# `git rebase`, git's "ours" / "theirs" labels are the OPPOSITE of what they
# mean during a `git merge`. A rebase replays the branch being rebased
# (feature/*, release/*, or hotfix/*, per docs/governance/GIT_WORKFLOW.md)
# commit-by-commit on top of its target (develop or main). Because of that
# replay direction, git calls the TARGET branch's content "ours" (stage 2)
# and the REPLAYED COMMIT's content -- the work actually being rebased --
# "theirs" (stage 3). Getting this backwards inverts every conclusion about
# "what each side was trying to accomplish," so this script labels both
# sides by role (target vs. incoming), not just by git's stage number, to
# make the correct reading unavoidable.
#
# Usage: bash show-conflict.sh <path>

set -euo pipefail

TARGET_PATH="${1:-}"

if [[ -z "$TARGET_PATH" ]]; then
  echo "Usage: show-conflict.sh <path>" >&2
  echo "Pass one repo-relative path that 'git status' currently lists as conflicted." >&2
  exit 1
fi

if [[ "$TARGET_PATH" == -* ]] \
   || ! [[ "$TARGET_PATH" =~ ^[A-Za-z0-9_][A-Za-z0-9_./-]*$ ]] \
   || [[ "$TARGET_PATH" == *..* ]]; then
  echo "Refusing to run: '$TARGET_PATH' is not a plain repo-relative path in the accepted" >&2
  echo "shape. No leading '-', no '..' segments, no leading '/', restricted character set." >&2
  echo "This script only accepts a bare path exactly as it appears in 'git status' output." >&2
  exit 1
fi

cd "$(git rev-parse --show-toplevel)"

CONFLICTED="$(git status --porcelain=v1 -- "$TARGET_PATH" \
  | awk '$1 ~ /^(DD|AU|UD|UA|DU|AA|UU)$/ {found=1} END {print found+0}')"

if [[ "$CONFLICTED" != "1" ]]; then
  echo "Refusing to run: '$TARGET_PATH' is not currently listed as an unmerged/conflicted" >&2
  echo "path by 'git status'. This script only inspects paths that are actually mid-conflict" >&2
  echo "right now -- run rebase-state.sh first to see the current list of conflicted paths." >&2
  exit 1
fi

echo "== Conflict sides for: $TARGET_PATH =="
echo "(remember the rebase 'ours'/'theirs' flip explained in this script's header comment)"
echo

echo "--- common ancestor / base (stage 1), if present ---"
git show ":1:$TARGET_PATH" 2>/dev/null \
  || echo "(no base version -- this path was added independently on one or both sides)"
echo

echo "--- TARGET branch content -- what you are rebasing ONTO (git calls this 'ours', stage 2) ---"
git show ":2:$TARGET_PATH" 2>/dev/null \
  || echo "(no version on this side -- this path was deleted here)"
echo

echo "--- INCOMING commit content -- the work being replayed (git calls this 'theirs', stage 3) ---"
git show ":3:$TARGET_PATH" 2>/dev/null \
  || echo "(no version on this side -- this path was deleted here)"
echo

echo "--- working-tree file as it stands right now, with conflict markers ---"
cat -- "$TARGET_PATH" 2>/dev/null || echo "(could not read working-tree file)"
