#!/usr/bin/env bash
# diverge-summary.sh <target-branch> [<source-branch>]
#
# Read-only helper for the smart-rebase skill. Summarizes how far a branch
# has diverged from the branch it would be rebased onto, BEFORE any rebase
# is started -- so smart-rebase can judge scale and risk (many commits on
# both sides, heavy file overlap) and set expectations about how much
# hunk-by-hunk reasoning a rebase is likely to need, or flag upfront that
# the histories look too far apart for confident automated reasoning.
#
# <target-branch> is the rebase target per docs/governance/GIT_WORKFLOW.md's
# branch model: 'develop' for a feature/* branch, 'main' for a release/* or
# hotfix/* branch. <source-branch> defaults to the current branch (HEAD) if
# omitted.
#
# WHY THE VALIDATION IS STRICT: both branch names are checked against a
# plain-branch-name shape (no leading '-', no '..' segment) AND confirmed to
# resolve to a real ref via 'git rev-parse --verify' before either one is
# used in a revision range. That combination means neither argument can ever
# be interpreted by git as a flag (a real ref never starts with '-') and
# neither can be a free-form string smuggling something else in -- git
# rev-parse --verify fails closed on anything that isn't a genuine existing
# ref. No '--output' (or any other) flag can reach git through this script.
#
# Usage: bash diverge-summary.sh <target-branch> [<source-branch>]

set -euo pipefail

BRANCH_SHAPE='^[A-Za-z0-9][A-Za-z0-9._/-]*$'

validate_branch_shape() {
  local b="$1"
  if [[ "$b" == -* ]] || ! [[ "$b" =~ $BRANCH_SHAPE ]] || [[ "$b" == *..* ]] || [[ "$b" == *.lock ]]; then
    echo "Refusing to run: '$b' is not a plain branch name in the accepted shape." >&2
    echo "No leading '-', no '..' segment, no leading '/', restricted character set." >&2
    exit 1
  fi
}

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  echo "Usage: diverge-summary.sh <target-branch> [<source-branch>]" >&2
  echo "  target-branch: 'develop' for a feature/* branch, 'main' for a release/* or hotfix/* branch" >&2
  echo "  source-branch: defaults to the current branch (HEAD) if omitted" >&2
  exit 1
fi
validate_branch_shape "$TARGET"

cd "$(git rev-parse --show-toplevel)"

SOURCE="${2:-}"
if [[ -z "$SOURCE" ]]; then
  SOURCE="$(git rev-parse --abbrev-ref HEAD)"
else
  validate_branch_shape "$SOURCE"
fi

# Fail closed unless both names resolve to real, existing refs.
if ! git rev-parse --verify --quiet "$TARGET" >/dev/null; then
  echo "Refusing to run: '$TARGET' does not resolve to an existing ref." >&2
  exit 1
fi
if ! git rev-parse --verify --quiet "$SOURCE" >/dev/null; then
  echo "Refusing to run: '$SOURCE' does not resolve to an existing ref." >&2
  exit 1
fi

MERGE_BASE="$(git merge-base "$TARGET" "$SOURCE")"

echo "== Divergence summary: '$SOURCE' vs. its rebase target '$TARGET' =="
echo
echo "Merge base: $MERGE_BASE"
git show -s --format='  %h  %ad  %s' --date=short "$MERGE_BASE"
echo

echo "== Commits on '$SOURCE' since the merge base (these are what would be replayed) =="
COMMITS_SOURCE="$(git log --format='%h  %ad  %s' --date=short --reverse "${MERGE_BASE}..${SOURCE}")"
[[ -n "$COMMITS_SOURCE" ]] && echo "$COMMITS_SOURCE" || echo "(none)"
echo

echo "== Commits on '$TARGET' since the merge base (what '$SOURCE' would be replayed onto) =="
COMMITS_TARGET="$(git log --format='%h  %ad  %s' --date=short --reverse "${MERGE_BASE}..${TARGET}")"
[[ -n "$COMMITS_TARGET" ]] && echo "$COMMITS_TARGET" || echo "(none)"
echo

echo "== Files touched on '$SOURCE' since the merge base =="
git diff --stat "${MERGE_BASE}..${SOURCE}"
echo

echo "== Files touched on '$TARGET' since the merge base =="
git diff --stat "${MERGE_BASE}..${TARGET}"
echo

echo "== Files touched on BOTH sides since the merge base (likely conflict candidates) =="
COMMON="$(comm -12 \
  <(git diff --name-only "${MERGE_BASE}..${SOURCE}" | sort) \
  <(git diff --name-only "${MERGE_BASE}..${TARGET}" | sort))"
[[ -n "$COMMON" ]] && echo "$COMMON" || echo "(none -- low risk of conflicts based on file overlap alone)"
