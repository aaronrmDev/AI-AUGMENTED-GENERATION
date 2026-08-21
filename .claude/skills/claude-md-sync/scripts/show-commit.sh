#!/usr/bin/env bash
# show-commit.sh <commit-hash> [path]
#
# Read-only helper for the claude-md-sync skill. Shows one commit (optionally
# scoped to one path), so a specific entry from find-sync-point.sh's commit
# list can be inspected before it's cited as evidence.
#
# WHY THIS SCRIPT EXISTS, AND WHY IT'S STRICT: git log, git show, and git diff
# all accept an --output=<file> flag that redirects the command's own output
# to an arbitrary file, overwriting it — functionally equivalent to handing
# the Write tool to whoever can construct the command line. git blame has an
# even sharper edge: --output=<file> silently truncates that file to zero
# bytes with no error, no confirmation, nothing. None of that is a crafted
# edge case; it's normal, documented git behavior. That is exactly the write
# path this skill's allowed-tools grant is built to close (see SKILL.md's
# hard-constraint section), which is why claude-md-sync never grants a raw,
# open-ended `git show ...` — only this wrapper, which validates its input
# before git ever runs, so no flag (--output or otherwise) can reach git no
# matter what text follows the script's path in the Bash permission grant.
#
# The commit argument must be a bare hex commit hash and nothing else — no
# flags, no revision expressions like HEAD~1 or branch names, no relative
# refs. That's deliberately narrow: the only legitimate use here is looking
# up a hash that find-sync-point.sh already printed, and a bare hex string
# can never be interpreted by git as an option, however it's spelled.
#
# Usage: bash show-commit.sh <commit-hash> [path]

set -euo pipefail

HASH="${1:-}"
PATHSPEC="${2:-}"

if ! [[ "$HASH" =~ ^[0-9a-fA-F]{4,40}$ ]]; then
  echo "Refusing to run: '$HASH' is not a bare hex commit hash (4-40 hex characters)." >&2
  echo "This script only accepts a commit hash in that exact shape -- no flags, no" >&2
  echo "revision expressions (HEAD~1, branch names, etc.), nothing else. Find the hash" >&2
  echo "you want from find-sync-point.sh's output and pass it as-is." >&2
  exit 1
fi

if [ -n "$PATHSPEC" ] && [[ "$PATHSPEC" == -* ]]; then
  echo "Refusing to run: the path argument '$PATHSPEC' starts with '-', which git could" >&2
  echo "interpret as a flag instead of a path. Pass a plain repo-relative path." >&2
  exit 1
fi

cd "$(git rev-parse --show-toplevel)"

if [ -n "$PATHSPEC" ]; then
  git --no-pager show --no-color "$HASH" -- "$PATHSPEC"
else
  git --no-pager show --no-color "$HASH"
fi
