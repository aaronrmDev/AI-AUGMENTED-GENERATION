# Git and Documentation Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring every document under `docs/`, plus `CLAUDE.md` and `README.md`, into agreement with the repository as it stands at `develop` 0f31cb3. Then publish `develop`, which is 80 commits ahead of `origin/develop`.

**Architecture:** This is a documentation-only branch, `feature/git-docs-sync`, cut from `develop` at 0f31cb3.
- **Task 1** fixes two known governance drifts: the feature-merge convention, and the hotfix release count.
- **Tasks 2 to 5** each audit one area of the documentation against the code, migrations, tests, reports, and git history. Each fixes only claims that evidence contradicts.
- **Controller steps:** after the final review, the controller merges the branch into `develop`, rehearses CI, pushes `develop`, and confirms that GitHub Actions succeed.

**Tech Stack:** Markdown, Mermaid, git, the repo-hygiene GitHub Actions workflow, and pytest collection for test counts.

**Spec:** None, because this is maintenance rather than a feature. The binding authority is the repository itself: its git history, `src/`, `alembic/`, `tests/`, `evaluation/reports/`, `.github/`, and `.claude/skills/`. Rulings made here are therefore judged against that evidence.

## Global Constraints

- Work only in the worktree `F:/Project/AI-AUGMENTED-GENERATION/.worktrees/feature/git-docs-sync`. Never edit the main checkout at `F:/Project/AI-AUGMENTED-GENERATION`, which has `develop` checked out.
- Every changed claim must trace to evidence named in the task report: a `file:line`, a commit hash, or a command with its output. Change nothing on a hunch. When the evidence says a claim is accurate, leave it alone.
- `docs/governance/WRITING_STANDARDS.md` binds every edit:
  - connected prose over labeled checklists;
  - open with what is true, never with a negation;
  - every claim needs a trace;
  - write for a colleague who just joined.
- Match the surrounding voice, sentence length, and density. Correct claims in place, and don't restructure or re-style a section beyond what the correction needs.
- Editable files are `CLAUDE.md`, `README.md`, `docs/**` (excluding the two directories below), and `.claude/skills/smart-rebase/SKILL.md` (Task 1 only).
- Read-only (never edited):
  - `docs/inputs/concepts/**`, the source extractions;
  - `docs/superpowers/**`, the historical SDD scaffolding; only the controller edits this plan's execution notes;
  - `evaluation/reports/**`, point-in-time measurement records;
  - `src/**`, `tests/**`, `alembic/**`, `.github/**`, and `docker/**`.
  - If the code rather than a doc is wrong, record it in the report as a code finding and don't fix it.
- CI hygiene (`.github/workflows/repo-hygiene.yml`) must stay green, and the rehearsal script below reproduces all three jobs:
  - No all-caps `TBD`, `TODO`, or `FIXME` token, and no occurrence of the word "placeholder" in any case, in `CLAUDE.md`, `README.md`, or `docs/**/*.md`. The two exceptions are `docs/inputs/concepts/` and `docs/superpowers/`.
  - Every `(docs/…md)`, `(.github/…)`, `(.gitmessage)`, and `(.githooks/…)` link in `CLAUDE.md` and `docs/README.md` resolves to a file.
  - No API-key-shaped value and no tracked `.env`.
  - Before every commit, run `bash .superpowers/sdd/2026-09-14-git-docs-sync/ci-rehearsal.sh` from the worktree root. It must print three OK lines.
- Commits:
  - Use Conventional Commits with the `docs:` type, following `.gitmessage`.
  - The body says what drifted and the evidence.
  - The last line is `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
  - Write the message to a file and use `git commit -F <file>`. Never pipe it to `-F -`.
  - One commit per task. Don't amend, rebase, merge, push, or touch any other branch.
- Preserve LF line endings and UTF-8 encoding.
- Implementers and reviewers never dispatch subagents.

### Verified facts every task can rely on

These were verified by the controller on 2026-09-14 at `develop` 0f31cb3.

- **Tests:**
  - 1,171 unit tests across 156 files in `tests/unit/`.
  - 285 integration tests across 65 files in `tests/integration/`. The last full run passed 277 and skipped 8, the vLLM tests that need the project's GPU machine.
  - mypy strict is clean on 232 source files.
  - These are the numbers `TESTING.md`, `README.md`, and `CLAUDE.md` already state.
- **Tags:**
  - `v1.0.0` (2026-09-03) is the one release, a merge of `release/1.0.0` into `main`.
  - `v1.0.1` to `v1.0.5` (2026-09-03 to 2026-09-06) are five hotfixes. Each landed on `main` as a real merge commit named `merge: hotfix/<name> -- …`, and each hotfix commit is also reachable from `develop`.
  - `v1.0.5` is `main`'s tip, dfc7802.
  - No GitHub Release objects exist, only tags.
- **`main`'s history:** it is linear before Gitflow (45 single-parent commits, none with a PR-number suffix). Since Gitflow it receives only real merge commits: one release and five hotfixes.
- **`develop`'s first-parent history:**
  - Finished feature work lands as real merge commits named `merge: feature/<n>-<desc> -- <summary>`, or `merge: <batch>` before the `feature/` prefix was used.
  - Every feature merge since `v1.0.5` (commit cec060b onward) is a real merge commit: 18 merges, and 3 small commits made directly on `develop` (84db3d5, 6c33136, cc329e4).
  - Before `v1.0.0`, the first-parent line holds 11 merge commits and 162 direct commits.
  - No squash merge is identifiable anywhere in the history.
- **The HTTP surface:**
  - `POST /auth/register`, `/auth/login`, `/auth/refresh`, and `/auth/logout`;
  - `POST /chat`, the RAG-only path;
  - `POST /sessions`, `GET /sessions`, and `POST /sessions/{session_id}/answers`, the unified per-query path.
  - No ingestion endpoint exists, and no frontend, workers, or Kubernetes manifests exist.
- **Open GitHub issues:** Epic #182 (Unified API) and Task #193 (the security hardening follow-up).

### Preflight rulings

- **Ruling P1:** the documentation changes to describe real merge commits for `feature/*` into `develop`, rather than practice changing to squash.
  - **Why:** the history cannot be rewritten, and a squash would make every task commit unreachable once the branch is deleted. Plans' execution notes and issue closing comments cite those task commits by hash.
  - **Cost if wrong:** if the owner wants squash, it is a docs-only revert plus squashing future features.
- **Ruling P2:** cutting a `release/*` into `main` and tagging it is out of this plan's scope. It is a publish, which needs the owner's go-ahead, so the controller asks after `develop` is published. If a release follows, its own branch updates `GIT_WORKFLOW.md`'s "Where this stands today".
  - **Cost if wrong:** one extra round-trip before `main` catches up.
- **Ruling P3:** evaluation reports are point-in-time records, so a report that describes the system as it was when it was measured isn't drift. Only a doc that presents such a statement as the current state is.
  - **Cost if wrong:** a stale sentence inside a dated report survives.

---

### Task 1: Feature-merge convention and release history in the governance docs

**Files:**
- Modify:
  - `docs/governance/GIT_WORKFLOW.md`, the opening paragraph (line 3) and lines 19–31;
  - `CLAUDE.md` line 49 ("Git-native workflow") and line 116 (the doc-map entry for `GIT_WORKFLOW.md`);
  - `docs/README.md` line 50;
  - `docs/governance/KANBAN_AUTOMATION.md` line 34;
  - `docs/governance/INTELLIGENT_REBASE.md` lines 3 and 7;
  - `docs/architecture/OVERVIEW.md` line 154, the phrase "squash-versus-merge ruling";
  - `.claude/skills/smart-rebase/SKILL.md` line 3, the frontmatter `description`. It must stay valid YAML on a single line.

**Interfaces:**
- **Consumes:** the verified facts above.
- **Produces:** the settled wording later tasks must not contradict. `feature/*` branches merge into `develop` with a real merge commit, and so do `release/*` and `hotfix/*` into both `main` and `develop`. The hotfix lifecycle has shipped five patch releases, `v1.0.1` to `v1.0.5`.

- [ ] **Step 1: Confirm the evidence yourself**

Run from the worktree root:
```bash
git log --first-parent --merges --format='%h %s' develop | head -30
git log --first-parent --no-merges --format='%h %s' cec060b..develop
git for-each-ref refs/tags --sort=creatordate --format='%(refname:short) %(creatordate:short) %(subject)'
git log --first-parent --format='%h %p %s' main | head -8
grep -rnoE '\b[0-9a-f]{7}\b' docs/superpowers/plans/2026-09-13-unified-api-sessions.md | head -5
```
Expected results:
- `develop` shows `merge: feature/…` commits;
- six tags, the five after `v1.0.0` being hotfixes;
- `main` shows two-parent `merge: hotfix/…` commits;
- the #183 plan cites commit hashes.

- [ ] **Step 2: Rewrite `GIT_WORKFLOW.md`'s ruling to match practice**

Replace the squash ruling (lines 21–23) with the ruling actually practiced: every branch type lands with a real merge commit.
- Keep the part of the reasoning that still holds: real merge commits are what make `main`'s tags traceable to their release or hotfix branch.
- Give the reason it extends to `feature/*` into `develop`:
  - The `merge: feature/<n>-<desc> -- <summary>` commit on `develop`'s first-parent line already reads as one entry per finished feature. That is the readability squashing was meant to buy.
  - Keeping the real merge preserves the task commits that plans' execution notes and issue closing comments cite by hash. A squash would orphan them once the branch is deleted.
- Say plainly that this revises the squash-merge ruling this document first made, and that it matches the repository's own history since Gitflow was adopted. Don't pretend the earlier ruling never existed.

Also make these edits:
- **Line 3:** the opening sentence says the doc reconciles Gitflow "with this project's squash-merge-to-main convention". Restate it so it doesn't claim a squash convention is in force.
- **Line 27 ("Where this stands today"):**
  - Replace "squash-merged back into it" with the real merge-commit wording.
  - Replace "exercised twice since … shipped as v1.0.1 and v1.0.2" with five hotfixes shipped as `v1.0.1` to `v1.0.5`.
  - Bring the batch inventory up to date after the first release: CAG's remaining techniques and combinations, the orchestration meta-layer batches, and the unified API endpoint all landed through `feature/*` branches merged into `develop`.
  - Name `develop`'s current lead over `main` as 107 commits, with no release cut since `v1.0.0`.
- **Line 31:** check it and keep it if accurate.

- [ ] **Step 3: Align every other statement of the rule**

Apply the same correction wherever the squash convention is stated as current:
- **`CLAUDE.md:49`:** "adopted alongside this project's existing squash-merge-to-main convention rather than in place of it", and "the ruling that reconciles Gitflow's merge-commit convention with squash-merging".
- **`CLAUDE.md:116`:** "how it reconciles with this project's squash-merge-to-main convention".
- **`docs/README.md:50`:** "the ruling that reconciles Gitflow's merge-commit convention with this project's existing squash-merge-to-main rule".
- **`KANBAN_AUTOMATION.md:34`:** "ending in a local squash-merge to `develop` with no PR ever opened". Keep the point that no PR is opened.
- **`INTELLIGENT_REBASE.md:3,7`:** "which of those merges squashes and which keeps real merge commits", and "squash-merging a finished `feature/*` branch into `develop`".
- **`OVERVIEW.md:154`:** "`docs/governance/GIT_WORKFLOW.md`'s squash-versus-merge ruling". Rename it to what the ruling now is, for example "merge-commit ruling".
- **`.claude/skills/smart-rebase/SKILL.md:3`:** "the final squash-merge/merge-commit step". Make it "the final merge-commit step", leaving the rest of the description untouched.

Keep each sentence's other content intact.

- [ ] **Step 4: Verify that nothing still states squash as current practice**

Run:
```bash
grep -rniE 'squash' CLAUDE.md README.md docs --include='*.md' | grep -vE 'docs/(inputs/concepts|superpowers)/'
grep -niE 'squash' .claude/skills/smart-rebase/SKILL.md
python -c "import yaml,io; t=open('.claude/skills/smart-rebase/SKILL.md',encoding='utf-8').read().split('---')[1]; print(sorted(yaml.safe_load(t)))"
bash .superpowers/sdd/2026-09-14-git-docs-sync/ci-rehearsal.sh
```
Expected results:
- every remaining "squash" occurrence is historical, meaning it describes the revised earlier ruling;
- the YAML parses and lists the same keys as before;
- three OK lines.

If `python` lacks `yaml`, use `F:/Project/AI-AUGMENTED-GENERATION/.venv/Scripts/python.exe`.

- [ ] **Step 5: Commit**

```bash
git add docs/governance/GIT_WORKFLOW.md CLAUDE.md docs/README.md docs/governance/KANBAN_AUTOMATION.md docs/governance/INTELLIGENT_REBASE.md docs/architecture/OVERVIEW.md .claude/skills/smart-rebase/SKILL.md
git commit -F <message-file>
```
Subject: `docs: record the real merge-commit convention and five hotfix releases`.

---

### Audit method shared by Tasks 2–5

Each audit task repeats this method in full, so an implementer reading only one task has all of it.

1. Read each assigned document end to end.
2. For every statement about this repository's current state, verify it against the repository. Such statements include:
   - a file, module, class, function, or test path;
   - a count;
   - a built or unbuilt status;
   - a migration, table, column, index, or RLS policy;
   - a route or status code;
   - an environment variable or default value;
   - a report path or a measured number attributed to a report;
   - a skill name;
   - a cross-reference to another doc's section.
3. Use `Glob`, `Grep`, `Read`, `git log`, and `git ls-files`. For class and function inventories, use Python's `ast` module.
4. Leave three kinds of statement alone:
   - statements about the concept sources or external tools in general;
   - narrative explicitly framed as past, such as "at the time" or "this generation used";
   - design intent explicitly labeled as not yet built.
5. Fix a statement only when the evidence contradicts it. The fix states the verified current fact in the document's own voice. If a statement is out of date because a later batch built the thing it calls unbuilt, update the status and point at the real code or report.
6. Write the report. Its findings list gives, for each finding, the location, the old claim, the evidence, and the new text. Its coverage list names every section checked and the evidence used, including sections found accurate.
7. If a task finds nothing to fix, it makes no commit and its report says so.

---

### Task 2: Audit `OVERVIEW.md` and `CONTEXT_GRAPH.md`

**Files:**
- Audit and modify: `docs/architecture/OVERVIEW.md` and `docs/architecture/CONTEXT_GRAPH.md`.
- Evidence sources:
  - `src/` (all bounded contexts, and `src/api/` including routers, `dependencies.py`, `unified_pipeline.py`, and `rate_limit.py`);
  - `alembic/versions/`;
  - `tests/`;
  - `evaluation/reports/`;
  - `docker/`;
  - `pyproject.toml`.

**Interfaces:**
- **Consumes:** Task 1's merge-commit wording, which must not be contradicted, and the verified facts.
- **Produces:** the corrected architecture statements Tasks 3–5 cross-reference.

- [ ] **Step 1: Apply the shared audit method to both files**

Follow the audit method above in full. The two files need these specific checks.

`OVERVIEW.md`:
- **Phase 1 module blueprint:** every path it marks as built exists, and every path it marks as unbuilt is truly absent.
- **Orchestration meta-layer section:** its descriptions of the Paradigm Router, the Context Budget Allocator (the 128K budget split into five slices), the Latency-Adaptive Fallback Cascade, and the Freshness-Aware Data Router match `src/orchestration/`.
- **Session endpoints paragraph:** it matches `src/api/routers/sessions.py`, and says the API cascade wires MAG and RAG only.
- **Stack and served-models statements:** they match `pyproject.toml` and `docker/docker-compose.yml`, wherever they state what the repo uses rather than what the brief proposed.

`CONTEXT_GRAPH.md`:
- Every Module node names a directory or file that exists.
- Each class count or class list stated for a node matches an `ast` count of that module's top-level classes.
- Every Test File node exists.
- The FUTURE or dashed nodes name only what is unbuilt: the ingestion HTTP endpoint, the frontend, workers, and Kubernetes.
- The "What isn't here yet" and "Keeping this current" sections agree with the diagram.
- Any total it states, such as "N real classes", is recounted.

A single command lists the top-level classes per module:
```bash
F:/Project/AI-AUGMENTED-GENERATION/.venv/Scripts/python.exe - <<'EOF'
import ast, pathlib
for p in sorted(pathlib.Path("src").rglob("*.py")):
    names = [n.name for n in ast.parse(p.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef)]
    if names: print(p.as_posix(), len(names), ",".join(names))
EOF
```

- [ ] **Step 2: Rehearse CI**

Run `bash .superpowers/sdd/2026-09-14-git-docs-sync/ci-rehearsal.sh`. Expected: three OK lines.

- [ ] **Step 3: Commit, only if something changed**

Subject: `docs: correct architecture overview and context graph drift`. The body lists each correction and its evidence.

---

### Task 3: Audit `RAG.md`, `CAG.md`, and `MAG.md`

**Files:**
- Audit and modify: `docs/architecture/RAG.md`, `docs/architecture/CAG.md`, and `docs/architecture/MAG.md`.
- Evidence sources: `src/rag/`, `src/cag/`, `src/mag/`, `src/orchestration/`, `tests/`, and `evaluation/reports/` (the measured numbers attributed to reports).

**Interfaces:**
- **Consumes:** the verified facts, and Task 2's commit (read `OVERVIEW.md` sections only if a paradigm doc cross-references them).
- **Produces:** accurate paradigm docs.

- [ ] **Step 1: Apply the shared audit method to all three files**

Follow the audit method above in full. The three files need these specific checks:
- **Every `src/…` and `tests/…` path cited:** the path exists.
- **Every measured number attributed to an `evaluation/reports/<name>.md` report:** the number appears in that report. Compare it exactly, including units.
- **Every "built", "measured", "validated against live vLLM", or "simulated" status:**
  - it matches the code and reports;
  - `CAG.md` says all nine techniques are built, four of them validated against a live vLLM server; verify both numbers against `evaluation/reports/cag-*.md`.
- **The RAG technique compatibility matrix and the MAG retrieval strategies:** the implementations they name exist.
- **Statements that RAG retrieval runs CPU work off the event loop:** they agree with `src/rag/application/search_documents.py`, `bm25_keyword_search.py`, the cross-encoder reranker, and `cache_warmed_retrieve.py`, which use `asyncio.to_thread`. Mention this only where the doc already discusses latency or concurrency; don't add new sections.

- [ ] **Step 2: Rehearse CI**

Run `bash .superpowers/sdd/2026-09-14-git-docs-sync/ci-rehearsal.sh`. Expected: three OK lines.

- [ ] **Step 3: Commit, only if something changed**

Subject: `docs: correct RAG, CAG, and MAG deep-dive drift`.

---

### Task 4: Audit `DATABASE.md`, `SECURITY.md`, `SECRETS_MANAGEMENT.md`, and `TESTING.md`

**Files:**
- Audit and modify:
  - `docs/database/DATABASE.md`;
  - `docs/security/SECURITY.md`;
  - `docs/security/SECRETS_MANAGEMENT.md`;
  - `docs/testing/TESTING.md`.
- Evidence sources:
  - `alembic/versions/` (0001–0007);
  - `src/*/infrastructure/` (repositories, Redis, Qdrant, and Neo4j adapters);
  - `src/api/`;
  - `src/identity/`;
  - `.githooks/pre-commit`;
  - `.env.example`;
  - `.github/workflows/repo-hygiene.yml`;
  - `tests/conftest.py`, `tests/unit/`, and `tests/integration/`;
  - `pyproject.toml` (pytest markers and configuration).

**Interfaces:**
- **Consumes:** the verified facts, including the test counts.
- **Produces:** accurate data, security, and testing docs.

- [ ] **Step 1: Apply the shared audit method to all four files**

Follow the audit method above in full. The four files need these specific checks.

`DATABASE.md`:
- Every table, column, type, constraint, index, and RLS policy it states for PostgreSQL matches migrations 0001–0007.
- Every Qdrant collection name and payload index, every Redis key pattern (`chat:{user_id}`, `sessions:{user_id}`, `{route}:{client_ip}`, the refresh-token keys, the MAG working-memory keys), and every Neo4j label and relationship matches the adapters that create them.

`SECURITY.md`:
- Every one of the 20 controls that it marks as implemented or verified points at real code, a test, a hook, or a workflow.
- Every control marked as planned is actually absent.
- The rate-limit paragraph matches `src/api/rate_limit.py` and `src/identity/infrastructure/redis_rate_limiter.py`: a fixed window of 60 s; auth 5/min per client IP; chat and answers sharing 100/min per user via `CHAT_RATE_LIMIT_PER_MINUTE`; session creation 20/min.
- The JWT lifetimes and Argon2id parameters match `src/identity/`.
- Its references to #193 match the open issue.

`SECRETS_MANAGEMENT.md`:
- The patterns and exclusions it describes match `.githooks/pre-commit` and the workflow's secret-scan job.

`TESTING.md`:
- The unit and integration counts and file counts match the verified facts.
- The pyramid percentages follow from the file counts.
- Every test file it names exists.
- The performance targets it ties to cascade timeouts match `TierTimeouts` in `src/orchestration/`.
- The skip conditions it describes match the actual `pytest.mark.skipif` guards: the vLLM tests and the Ollama tests.

- [ ] **Step 2: Rehearse CI**

Run `bash .superpowers/sdd/2026-09-14-git-docs-sync/ci-rehearsal.sh`. Expected: three OK lines.

- [ ] **Step 3: Commit, only if something changed**

Subject: `docs: correct database, security, and testing doc drift`.

---

### Task 5: Audit the entry points, governance, evaluation docs, and ADRs

**Files:**
- Audit and modify:
  - `README.md`, `CLAUDE.md`, and `docs/README.md`;
  - `docs/governance/SKILL_ROUTING.md`, `AUTOLEARNING.md`, `TOKEN_ECONOMY.md`, `WRITING_STANDARDS.md`, and `KANBAN_AUTOMATION.md`;
  - `docs/evaluation/COMPARISON_METHODOLOGY.md`, `quantitative-template.md`, and `qualitative-rubric.md`;
  - `docs/decisions/adr/0001`–`0006`.
- Evidence sources:
  - `git ls-files`;
  - `.claude/skills/` (project skills);
  - `.github/ISSUE_TEMPLATE/` and `.github/PULL_REQUEST_TEMPLATE.md`;
  - `.gitmessage`;
  - `evaluation/`;
  - the output of Tasks 1–4 on this branch.

**Interfaces:**
- **Consumes:** every earlier task's commits, because the entry points summarize the docs those tasks corrected.
- **Produces:** entry points that agree with the corrected docs.

- [ ] **Step 1: Apply the shared audit method to every assigned file**

Follow the audit method above in full. The files need these specific checks:
- **`CLAUDE.md`'s documentation map and `docs/README.md`'s index:**
  - every file under `docs/`, apart from the two read-only directories, is listed exactly once;
  - no listed file is missing;
  - each one-line description agrees with that file's current content, including the corrections Tasks 1–4 made.
- **The summary paragraphs in `CLAUDE.md` and `README.md`:** built scope, test counts, and "what's still ahead" agree with the verified facts and with `OVERVIEW.md` and `CONTEXT_GRAPH.md` as Task 2 left them.
- **Every skill named in `SKILL_ROUTING.md` or `CLAUDE.md`'s skill-routing section:** it exists as a project skill under `.claude/skills/`, or is named as a user- or plugin-level skill. Project skills include `claude-md-sync` and `smart-rebase`.
- **`AUTOLEARNING.md`:** it agrees with `.claude/skills/claude-md-sync/SKILL.md` and its two scripts.
- **ADR status fields:** Accepted or Superseded matches the code. For example, ADR-0005 is superseded by 0006, and the code uses no LangChain.
- **The model roster in `COMPARISON_METHODOLOGY.md`:** where it states what the repo configures, as opposed to the intended roster, it matches `src/` provider settings and `.env.example`.
- **`KANBAN_AUTOMATION.md`:** keep the manual steps as documented intent. Correct only repository facts, such as template or workflow paths.

- [ ] **Step 2: Rehearse CI**

Run `bash .superpowers/sdd/2026-09-14-git-docs-sync/ci-rehearsal.sh`. Expected: three OK lines.

- [ ] **Step 3: Commit, only if something changed**

Subject: `docs: align entry points, governance, and ADRs with the corrected docs`.

---

## Controller steps after the final review

These steps are not dispatched.
1. Merge `feature/git-docs-sync` into `develop` in the main checkout with `git merge --no-ff -F <file>`, using a `merge: feature/git-docs-sync -- …` message.
2. On the merged result, rerun the CI rehearsal, the unit suite, and the doc link check.
3. Remove the worktree and delete the branch.
4. Push `develop` to `origin`. The owner asked for git to be completely up to date.
5. Confirm that the repo-hygiene run on the pushed head concludes success.
6. Record the outcome in the execution notes below.
7. Ask the owner whether to cut the `v1.1.0` release into `main` (Ruling P2).

## Execution notes

Executed inline via `superpowers:subagent-driven-development` in worktree `.worktrees/feature/git-docs-sync`, one fresh implementer subagent per task plus a dedicated task reviewer, and a fix loop where a review found issues. Full ledger: `.superpowers/sdd/2026-09-14-git-docs-sync/progress.md` (deleted with the workspace after the final review passed; the commits below are the durable record).

**Departures from the plan as written:**
- Task 4's implementer completed its work, committed (`b4a4b99`), and wrote its full report, but its session hit an Anthropic API rate limit before it could send its final short-status reply. The commit and report were verified complete and treated as the task's real submission; the session model was switched from Opus to Sonnet for the rest of the branch (later reverted to Opus for the final review once the rate limit reset), per the user's own `/model sonnet` mid-branch.
- Task 4's fix round 1 had to dispatch a fresh implementer rather than resume the original, since the original agent's session ended on the rate-limit error with no agent ID to resume — the fix loop's own fallback path for exactly this case.
- Ruling T4-K1 (made by the controller, applied through Task 4's fix loop): roughly 21 sentences in `SECURITY.md` and `TESTING.md` attributed a specific fact to "CLAUDE.md" that only a pre-2026-08-21 version of `CLAUDE.md` (before commit `2d187ca` rewrote it into a lean pointer file) actually contained. Ruled: drop the attribution, state the fact as the document's own content.
- Three items surfaced by earlier tasks' reviews as real but out of that task's file scope were explicitly routed to Task 5 rather than fixed piecemeal: the CAG "all nine built, eight measured" wording mirrored in `CLAUDE.md`, `docs/README.md`, `CONTEXT_GRAPH.md`, and `OVERVIEW.md`; `SECURITY.md`'s WebSocket-convention sentence describing a route that doesn't exist; and `TESTING.md`'s stale "browser-driven suite CLAUDE.md actually describes" attribution. All three were fixed by Task 5 and confirmed by its review.
- The final whole-branch review (Opus) found three Important and five Minor issues beyond what any task-scoped review could see — mostly claims that were consistent within one task's files but not across the whole tree (`docs/governance/INTELLIGENT_REBASE.md` still claiming `develop` didn't exist, `docs/architecture/RAG.md` citing a CLAUDE.md pipeline description that CLAUDE.md doesn't contain, and `docs/governance/WRITING_STANDARDS.md`'s worked example going stale relative to seven other corrected locations). All eight were fixed in one dispatch (`c3ecbd3`) and confirmed by one scoped re-review, per the skill's own final-review protocol.

**Commits (`feature/git-docs-sync`, `df4939b..c3ecbd3`, 11 commits):** the plan itself (`df4939b`); Task 1 merge-commit convention (`13770f3`); Task 2 architecture/context-graph audit plus one fix round (`5655807`, `b49fc19`); Task 3 RAG/CAG/MAG audit plus one fix round (`b8b9346`, `57b0dd2`); Task 4 database/security/testing audit plus one fix round (`b4a4b99`, `e3ca400`); Task 5 entry-points/governance/ADR audit, approved with no fix round (`e9cf297`); the final-review fix wave (`c3ecbd3`).

**Merge and push:** merged to `develop` locally as `6be02bc` (real merge commit, per the branch's own Task 1 finding). Unit suite (1,171 passed) and mypy (232 files clean) reran on the merged result; the CI hygiene rehearsal (link check, placeholder scan, secret scan) was green both on the branch and on the merged tree. Worktree and branch removed. Pushed to `origin/develop`; the `repo-hygiene` run on the pushed head (`6be02bc`) concluded `success`.

**Not done, by Ruling P2:** no `release/*` branch was cut and no new tag was made. `develop` now leads `main` (`v1.0.5`) by 108 commits with no release since `v1.0.0`; cutting `v1.1.0` is a decision for the repository owner, not this plan.
