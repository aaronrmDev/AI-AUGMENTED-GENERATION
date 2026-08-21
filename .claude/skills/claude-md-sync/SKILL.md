---
name: claude-md-sync
description: Checks whether CLAUDE.md and the docs/ tree have drifted out of sync with what the repository actually contains, by comparing their current claims against git history since the last time either was touched. Use this whenever a user asks to check CLAUDE.md for staleness, sync the docs, verify the documentation still matches reality, or audit the doc map/ADRs/routing rules against recent commits. Also invoke it proactively, on your own judgment and without being asked, immediately after finishing any significant piece of work in this repository — standing up a new module or package, changing an architectural decision, reversing or superseding an ADR, or adding a new document the doc map doesn't yet account for — so that any claim CLAUDE.md or a linked doc makes about that area doesn't go unverified. Do not invoke it for trivial changes (typo fixes, formatting, one-line clarifications) that don't move the ground any doc is standing on. This skill is strictly read-only: it never edits CLAUDE.md, never edits anything under docs/, and never commits — it only produces a numbered list of proposed edits, each with a file, a claim, evidence, and a fix, for a human to review and apply by hand.
allowed-tools: Read, Grep, Glob, Bash(git ls-files:*), Bash(bash .claude/skills/claude-md-sync/scripts/find-sync-point.sh:*), Bash(bash .claude/skills/claude-md-sync/scripts/show-commit.sh:*)
---

# claude-md-sync

This skill implements the contract specified in `docs/governance/AUTOLEARNING.md`. If
anything here and that document ever disagree, the document wins — this file is one
faithful implementation of it, not a second source of truth. Read it if you want the
reasoning behind any rule below; this file gives you the operational steps.

## Why this exists

CLAUDE.md and everything under `docs/` describe this project's structure, rules, and
architecture. That description is only accurate for as long as nobody changes the thing it
describes, and code and docs drift apart silently, one ordinary commit at a time — a file
gets renamed and a doc-map entry keeps pointing at the old path, an ADR gets superseded in
practice before its status field catches up. Nobody decides that drift should happen; it
just accumulates because nothing is watching for it. This skill is that watch.

## The hard constraint — read this before doing anything else

**claude-md-sync proposes. It never applies.**

- Never use `Edit`, `Write`, or `NotebookEdit` on `CLAUDE.md` or on anything under `docs/`
  (or anywhere else) as part of this skill. Those tools are deliberately absent from this
  skill's `allowed-tools`, so this session cannot invoke them even if it wanted to.
- Never run a state-changing git command: no `git add`, `git commit`, `git push`,
  `git checkout` (write form), `git stash`, or anything else that changes the working tree,
  the index, or repo history. This skill's `allowed-tools` grants exactly three things:
  `git ls-files` (unscoped arguments — it hard-errors on anything resembling a write flag,
  confirmed empirically, not assumed), and two bundled scripts,
  `find-sync-point.sh` and `show-commit.sh`, invoked as exact commands. **Deliberately
  absent: any open-ended `git log`, `git show`, `git diff`, or `git blame` grant.** All four
  of those accept an `--output=<file>` flag that redirects their output to overwrite an
  arbitrary file — `git blame --output=<file>` even truncates the target file to zero bytes
  with no error — and a wildcard grant like `Bash(git show:*)` cannot exclude that flag; the
  trailing `*` lets it through no matter what else is intended. (This was tested and
  confirmed against this exact repo, not theorized — see the fix history in this skill's
  git log if you want the receipts.) That's why every place this skill needs to look at a
  specific commit routes through `show-commit.sh` instead: the script validates its
  argument as a bare hex commit hash *before* it ever builds a git command line, so
  `--output=...` (or any other flag) can't reach git regardless of what text follows the
  script's path in the Bash permission grant — the enforcement point is the script's own
  validation, not the permission wildcard. Don't invent a raw `git log`/`git show`/`git
  diff`/`git blame` call to route around this, and don't loosen either script's input
  validation to make an edge case more convenient — both defeat the reason this shape
  exists. If the job seems to need something the two scripts and `git ls-files` don't
  cover, that's a sign to stop and report the limitation, not to reach for an unscoped git
  command.
- Never hand off the writing step to something else that could do it unsupervised (another
  skill, a subagent, a background task). The output of this skill is always a list handed
  back to the human running this session — not an action taken on their behalf.
- If you notice yourself reasoning "I could just fix this one small thing while I'm here" —
  don't. A single unsupervised edit to CLAUDE.md, even an obviously-correct one, is the
  exact failure this skill exists to prevent, just moved up one level: the source of truth
  drifting from itself instead of from the repo. Report it as item N in the list instead.

This is not a soft preference weighed against convenience. Per the contract, if there is
ever tension between making this skill more automated or convenient and this constraint,
the constraint wins outright, with no exception.

## When to run it

On-demand only — there is no scheduler and no background trigger. It runs when:

1. **A human asks for it directly** — "check CLAUDE.md for staleness," "sync the docs,"
   "does the doc map still match reality," or similar.
2. **You (the agent) judge, on your own, that it's warranted**, immediately after finishing
   a piece of work. Apply this test: *did the work that was just completed change something
   CLAUDE.md or a linked doc makes a claim about?* If yes, that claim is unverified until
   this check runs. Concretely, that means: standing up a new module or package, changing
   an architectural decision (swapping a library, reversing or superseding an ADR, altering
   how two layers talk to each other), or adding a new document the doc map doesn't yet
   list. It does *not* mean every commit — a typo fix, a formatting pass, or a one-line
   clarification doesn't move the ground any doc is standing on, so don't invoke it for
   those.

## What it checks — the four inputs

There is no separate log of past runs. "Since the last known sync point" is anchored to
something git already knows: the most recent commit that touched `CLAUDE.md` or anything
under `docs/`, on the reasoning that the last time those files changed is the last time
someone actually reconciled them with reality. Gather exactly these four things, in order:

1. **The sync point** — the most recent commit touching `CLAUDE.md` or `docs/`.
2. **Everything since that commit** — the full list of commits after it, and what they
   changed (stats at minimum; full diffs for anything that looks doc-relevant).
3. **The current text of `CLAUDE.md`** — as it stands right now, not as of the sync point.
4. **The current shape of the `docs/` tree** — as it stands right now.

Run the bundled helper for steps 1 and 2 — it is read-only (see its header comment) and
saves you from re-deriving the same `git log`/`git diff` incantations every run:

```bash
bash .claude/skills/claude-md-sync/scripts/find-sync-point.sh
```

It prints the sync-point commit, the full list of commits since it (oldest first), and a
diffstat of everything that changed. For any commit in that list whose subject or file list
suggests it might contradict something CLAUDE.md or a doc claims, follow up with the other
bundled script to read the actual change — the diffstat alone tells you *that* something
changed, not *what* it now says:

```bash
bash .claude/skills/claude-md-sync/scripts/show-commit.sh <hash>
bash .claude/skills/claude-md-sync/scripts/show-commit.sh <hash> <path>   # scoped to one file
```

Use this script, not a raw `git show` — see the hard-constraint section above for why a raw
`git show` isn't available here at all. Pass the hash exactly as `find-sync-point.sh` printed
it; the script rejects anything that isn't a bare hex commit hash.

For steps 3 and 4, read `CLAUDE.md` directly and use `Glob`/`Read` to see the current
`docs/` tree — don't rely on memory of what either looked like earlier in the session, and
don't rely on the diffstat as a substitute for reading the current files: the diffstat shows
what changed, not what the documents assert *now*, and those can differ if a file changed
more than once since the sync point.

If the helper reports no sync point at all (nothing has ever touched `CLAUDE.md` or
`docs/`), or reports zero commits since the sync point, say so plainly in your output and
stop — there is nothing to check yet, and manufacturing findings in that situation would
violate the next section.

## What to look for

Against those four inputs, look for **concrete mismatches**, not a general sense that things
might have moved on. Three shapes of finding, per the contract:

- **A rule stated in CLAUDE.md or a governance doc that recent commits no longer follow in
  practice.** Example shape: a naming convention, a required checklist step, a stated
  process — check it against what the commits since the sync point actually did.
- **A doc-map or cross-reference entry naming a file that has since been renamed, moved, or
  deleted.** Check every file path CLAUDE.md or a `docs/` file points to against
  `git ls-files` reality.
- **An architecture or decision description that no longer matches what the corresponding
  code or document actually does now.** This includes ADR status fields (`Accepted` vs. a
  later commit message that records the team reversing the decision), pipeline or component
  descriptions, and stack/version claims.

A finding that can't be pinned to a specific file, a specific current claim, and a specific
piece of evidence isn't finished — investigate further until it's concrete, or drop it. Do
not report a vague hunch ("some of this might be outdated") dressed up as a finding.

## Output format

Produce a numbered list. Every entry has exactly four parts, in this order:

1. **Location** — the exact file and the location within it (section heading, table row,
   line reference) that the proposed edit applies to.
2. **Current claim** — what the document says right now, quoted or precisely paraphrased.
3. **Evidence** — the specific commit(s) that contradict it: hash, subject line, and (if not
   obvious from the subject) the specific change that creates the contradiction.
4. **Proposed fix** — specific replacement text or a specific addition. Not "update this" —
   the actual wording, or close enough to it that a human can approve it as-is.

Two illustrations of the required granularity (these are shape examples only, not real
findings — do not treat them as a checklist to search for verbatim in this repo):

> 1. `CLAUDE.md` doc map, skill-routing row — currently points to
>    `docs/governance/SKILL_ROUTING.md`, but commit `a1b2c3d` ("rename routing doc")
>    renamed that file to `docs/governance/ROUTING.md`. Update the pointer to
>    `docs/governance/ROUTING.md`.
> 2. `docs/decisions/adr/0002-qdrant-over-milvus.md` — status field currently reads
>    `Accepted`, but commit `9f8e7d6` ("chore: switch to Milvus for tenant sharding")
>    records the team reversing this decision. Change the status to `Superseded` and add a
>    follow-up ADR recording the reversal and its rationale.

If the check turns up nothing — the docs still match what the commits since the sync point
did — say that plainly: report the sync point, confirm how many commits were reviewed, and
state that no mismatches were found. Do not manufacture findings to look useful; a clean
result is a valid and useful result.

Close every run with a one-line reminder of the hard constraint, e.g.: *"These are proposals
only — nothing above has been written to any file. Tell me which items to apply, if any."*
