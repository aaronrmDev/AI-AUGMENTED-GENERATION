---
name: smart-rebase
description: Helps a human rebase a feature/*, release/*, or hotfix/* branch onto its Gitflow target (feature/* onto develop; release/* or hotfix/* onto main, per docs/governance/GIT_WORKFLOW.md) by reading both sides of every conflicting hunk, reasoning about what each side was trying to accomplish, and proposing a specific resolution with its reasoning -- instead of leaving raw conflict markers for a human to puzzle through alone. Use this whenever a user wants to rebase a branch onto develop or main, is stuck mid-rebase looking at <<<<<<< HEAD conflict markers, asks "help me resolve this rebase conflict," wants to know if a branch has drifted far from its target before starting a rebase, or is about to bring a long-lived feature/release/hotfix branch up to date with a target that moved on without it. Do NOT use this for a routine, conflict-free rebase (git handles that on its own with nothing to reason about), for an ordinary merge, or for the final squash-merge/merge-commit step that lands a finished branch into develop or main -- that step is separately specified in docs/governance/GIT_WORKFLOW.md and isn't a rebase at all. This skill only ever proposes conflict resolutions for a human to review; it never applies a resolution, never stages it, never runs `git rebase --continue`, and never runs `git push` in any form -- see the hard constraint below before assuming otherwise.
allowed-tools: Read, Grep, Glob, Bash(bash .claude/skills/smart-rebase/scripts/rebase-state.sh:*), Bash(bash .claude/skills/smart-rebase/scripts/diverge-summary.sh:*), Bash(bash .claude/skills/smart-rebase/scripts/show-conflict.sh:*)
---

# smart-rebase

This skill implements the contract specified in `docs/governance/INTELLIGENT_REBASE.md`. If
anything here and that document ever disagree, the document wins — this file is one
faithful implementation of it, not a second source of truth. Read it if you want the
reasoning behind any rule below; this file gives you the operational steps.

## Why this exists

`docs/governance/GIT_WORKFLOW.md` establishes this project's Gitflow branch model:
`feature/*` branches cut from `develop`, `release/*` and `hotfix/*` branches cut from and
merged back into `main`. Before any of those branches reach their final merge, they
sometimes need to be rebased onto a target that kept moving while they were being worked
on — and an ordinary `git rebase` that hits a conflict hands back raw `<<<<<<< HEAD` markers
with no opinion about which side is right. This skill closes that gap: for every conflicting
hunk it reads both sides in full, reasons about what each side was trying to accomplish, and
proposes a specific resolution with its reasoning, so a human reviewing the proposal can tell
whether it's right without re-deriving the answer from scratch.

## The hard constraint — read this before doing anything else

**smart-rebase proposes resolutions. It never applies them, never finalizes a rebase, and
never rewrites shared history — full stop, no exception for confidence or convenience.**

- Never run `git rebase --continue` (or `--skip`, or `--abort`) as an action this skill takes
  on its own judgment. Finishing a paused rebase is a decision the human makes, always.
- Never run `git push`, in any form — not a plain push, not `--force`, not
  `--force-with-lease`. Publishing rewritten history to anywhere another person could pull it
  from is exactly the hard-to-undo action this constraint exists to gate.
- Never apply a proposed resolution to a file, never run `git add`, and never start a rebase
  (`git rebase <target>`) on this skill's own initiative. All of that is why this skill's
  `allowed-tools` grants exactly `Read`, `Grep`, `Glob`, and three read-only wrapper scripts —
  **`Edit`, `Write`, `NotebookEdit`, and every other form of `Bash` are deliberately absent.**
  This isn't a scoped-down git wildcard the way `claude-md-sync`'s is — it's the complete
  absence of any tool this skill could use to write to a file, stage anything, or move a git
  ref. A session running this skill cannot mutate the repository even if it tried, because
  nothing in its tool grant is capable of it. That's a deliberately stronger posture than
  `claude-md-sync`'s (which still needed `Bash(git ls-files:*)` for a harmless read), because
  a rebase gone wrong is a fundamentally worse failure than a stale doc: `claude-md-sync`'s
  worst case is a wrong sentence sitting in a file until someone corrects it; this skill's
  worst case, if it could act unsupervised, would be rewritten commit history — and once
  that's force-pushed over a branch other people have pulled from, undoing it cleanly is a
  genuinely hard problem, not a one-line fix. Reflogs expire, collaborators' local branches
  diverge further with every commit they make on top of the bad state, and a plain revert
  doesn't cleanly apply to history that's already been rewritten and shared elsewhere. That
  asymmetry — a wrong doc edit is trivially reversible, a bad rewrite may not be — is why this
  skill's tool grant removes the capability entirely instead of trying to scope it carefully.
- **On shell redirection (`>`, `>>`, `2>`) — what's proven versus what's inferred.** Every
  exploit attempt actually run against these scripts during this skill's build and its review
  (flag injection via `--output=`, path traversal, `$(...)`/`;`/`&&` shell-metacharacter
  injection, non-conflicted-path bypass, nonexistent-ref bypass) was run from a session with
  broader-than-`smart-rebase` permissions, because there is no way to launch a session
  genuinely restricted to this skill's own `allowed-tools` and then attack it from the
  outside. That means one specific angle — `bash .../rebase-state.sh > some/file` or `2>` or
  `>>` redirecting a wrapper script's own stdout/stderr onto an arbitrary path — has never
  been empirically exercised against a truly restricted session, by anyone who has worked on
  this skill. What's true instead is documented: current Claude Code documentation describes
  output redirection as gated through a separate mechanism from the subcommand decomposition
  that blocks `&&`/`;`/`|` chaining — a redirect is checked as a file write against `Edit`
  allow/deny rules, not against the `Bash` grant alone, and `smart-rebase` grants zero `Edit`
  permission of any kind (see the point above). If that documentation is accurate, a redirect
  attempt should require explicit human approval rather than silently succeeding. That is
  **documented-as-safe, not tested-as-safe**, and this file says so plainly rather than
  either overclaiming a verified guarantee or quietly leaving the gap unmentioned — the same
  citation discipline `docs/governance/WRITING_STANDARDS.md` requires of every claim in this
  repository: trace it to a test, a source, or flag it as unverified, never assert it bare.
- The three wrapper scripts this skill can run — `rebase-state.sh`, `diverge-summary.sh`, and
  `show-conflict.sh` — are read-only by construction and validate every argument before it
  ever reaches git (see each script's own header comment for the specific exploit it closes,
  modeled directly on `claude-md-sync/scripts/show-commit.sh`'s bare-hex-hash validation). Do
  not invent a raw `git rebase`, `git add`, `git push`, `git commit`, `git checkout`, or
  `git reset` call to route around the absence of those tools, and do not loosen any script's
  validation to make an edge case more convenient — both defeat the reason this shape exists.
- Never hand the writing or finishing step off to something else that could do it
  unsupervised (another skill, a subagent, a background task). Every proposal this skill
  produces goes back to the human running the session — never an action taken on their
  behalf.

Per the contract, if there is ever tension between making this skill faster or more automated
and this constraint, the constraint wins outright, with no exception.

## What happens after a proposal is approved

Because this skill cannot write anything, **the human is the one who does every subsequent
step themselves, in their own shell, outside this skill's tool grant**: editing the
conflicted file to match an approved resolution, running `git add` on it, running
`git rebase --continue` once every conflict in the current step is resolved, and — only once
the whole rebase is finished and the human has decided to publish it — running `git push`
(force, or force-with-lease, since a rebase rewrites commits that may already be on the
remote). This skill's job ends at handing back a clear, reasoned proposal; picking it up from
there is always a deliberate human action, never an automatic next step this skill takes.

## When to run it

On-demand only — there is no scheduler and no background trigger. Two moments call for it:

1. **Before starting a rebase**, to gauge scale and risk. Run `diverge-summary.sh` with the
   Gitflow target as its first argument (`develop` for a `feature/*` branch, `main` for a
   `release/*` or `hotfix/*` branch) to see the merge base, the commits and files each side
   has touched since it, and — most useful for setting expectations — the files touched on
   *both* sides, which are the likely conflict candidates. A summary showing many commits on
   both sides with heavy file overlap is a signal to expect several hunks and possibly lower
   confidence on some of them; say so up front rather than letting that surface as a surprise
   partway through.
2. **While a rebase is paused on a conflict**, after the human has already run
   `git rebase <target>` themselves (this skill never starts one) and git has stopped with
   conflict markers in one or more files. Run `rebase-state.sh` first to confirm a rebase is
   actually in progress and see the full list of conflicted paths, then `show-conflict.sh
   <path>` for each one.

## What "intelligent" means here, concretely

For each conflicted path, `show-conflict.sh <path>` prints four things: the common-ancestor
version (if one exists), the target branch's version, the incoming commit's version, and the
working-tree file as it stands right now with conflict markers. **Read the script's own
header comment on the "ours"/"theirs" rebase flip before reasoning about which side is
which** — during a rebase, git calls the *target* branch's content "ours" and the *replayed
commit's* content "theirs," which is the reverse of what those words mean during a merge, and
getting it backwards inverts every conclusion about intent. With both sides in hand, work out
what each one was actually trying to accomplish: a rename one side made that the other side's
edit doesn't know about, a bugfix layered onto code the other side refactored around, two
independent additions to the same list that both belong in the result, or a genuine
disagreement about what the code should do. Then write a specific proposed resolution — the
actual merged text, not a description of one — together with the reasoning that produced it,
in enough detail that the human reviewing it can tell whether the reasoning is right without
re-deriving the answer themselves. That per-hunk reasoning is the entire point: a blanket
`--strategy-option=ours`/`theirs` discards one side uniformly, where two conflicts three lines
apart in the same file can easily need opposite resolutions.

Present each conflicted path as its own numbered item, in this shape:

1. **Path** — the conflicted file.
2. **What each side did** — a short, concrete summary of the target branch's change and the
   incoming commit's change, in terms of intent, not just a restatement of the diff.
3. **Proposed resolution** — the actual resolved text for the hunk (or the whole file, if
   that reads more clearly), not a vague instruction.
4. **Reasoning** — why that resolution is the right synthesis of both sides' intent.
5. **Confidence** — state it plainly: confident, or not — and if not, say what's genuinely
   ambiguous rather than presenting a guess with the same certainty as a clear-cut case.

## What to do when you aren't confident

Some hunks don't resolve into a clean story about intent — the two changes contradict each
other at the level of what the code should do, not just how it's phrased, and no amount of
re-reading turns that into a confident answer. When that happens, say so plainly in the
numbered item above and stop there — do not pick a side and present it with the same
confidence as a clear-cut case. **A wrong resolution is worse than no resolution**: an
unresolved conflict is visibly unresolved and stops the rebase where anyone can see it, where
a wrong resolution that reads smoothly can slip through review and land silently in `develop`
or `main`. The same applies to a whole rebase, not just one hunk — if `diverge-summary.sh`
or the accumulating pattern of conflicts suggests the two histories have diverged too far for
hunk-by-hunk reasoning to be trustworthy (a long-abandoned branch, a target that's been
substantially restructured since the branch split), say that too, rather than working through
every remaining hunk at declining confidence and presenting all of them as equally solid.

## The three wrapper scripts

- `bash .claude/skills/smart-rebase/scripts/rebase-state.sh` — no arguments. Reports the
  current branch, whether a rebase is in progress and what it's rebasing onto, and the full
  list of currently conflicted paths. Run this first whenever you're picking up a paused
  rebase.
- `bash .claude/skills/smart-rebase/scripts/diverge-summary.sh <target-branch>
  [<source-branch>]` — `target-branch` is `develop` or `main` per `GIT_WORKFLOW.md`;
  `source-branch` defaults to the current branch. Read-only look at how far the two branches
  have diverged, before a rebase is even started.
- `bash .claude/skills/smart-rebase/scripts/show-conflict.sh <path>` — `path` must be a
  bare, repo-relative path that `git status` currently lists as conflicted (the script
  verifies this itself and refuses otherwise). Shows the base, target, and incoming versions
  of that path, labeled by role to avoid the "ours"/"theirs" rebase flip.

Use these, not a raw `git log`/`git show`/`git diff`/`git blame`/`git rebase` call — those
tools aren't in this skill's `allowed-tools` at all, for the reasons the hard-constraint
section above explains in full.
