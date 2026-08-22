# Secrets Management Protocol

`docs/security/SECURITY.md` states, as point 1 of the 20-point checklist, that API keys stay in environment variables and never in code — this document is what makes that a checkable procedure for this project's actual secrets (the Anthropic, Gemini, and DeepSeek API keys behind `docs/evaluation/COMPARISON_METHODOLOGY.md`'s model roster, and the local vLLM endpoint URL) rather than a general principle nobody has to act on. Two layers, on purpose: a written protocol for what a human is supposed to do, and a mechanical check that runs whether or not anyone remembers to do it — because "supposed to" is exactly the kind of unverified claim this project's writing standard doesn't let stand on its own.

## Where each key lives, and why nowhere else

Every real secret lives in `.env` at the repository root, loaded into the running process as an environment variable, and nowhere else — never hardcoded into a script, never pasted into a commit message, never logged, and never pasted into a chat session, an issue, or a PR description, all of which persist in places this project doesn't control the retention of. `.env` is listed in `.gitignore` (verified working via `git check-ignore .env` — it returns a match) and has never been committed to this repository's history (verified via `git log --all --full-history -- .env` — empty). `.env.example` is the committed counterpart: variable names only, no real values, so the actual secret-loading convention lives in the repository even though no secret does.

## Loading protocol

Read every key through the environment at process start (`os.environ` / `process.env` / the framework's settings loader), never by parsing `.env` ad hoc inside application logic — a dedicated loader (e.g. `python-dotenv` for local development, real environment injection in any deployed environment) keeps the parsing in one place instead of scattered wherever a script happens to need a key. Never write a key to a log line, an error message, or a debug print — an exception that includes the request headers or a config dump is a common accidental leak path, and the fix is to redact known-sensitive keys (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`) at the logging layer itself, not to trust that nobody ever logs the wrong object.

## Least privilege per provider

Scope each key to only what this project actually calls, and revisit the scope when a provider's console allows finer-grained keys than it did when the key was created — a key that has broad account permissions but is only ever used for `messages.create`-style calls is carrying risk it doesn't need to. This is a "design intent, not yet verified" item as of this writing: none of the three provider consoles' current fine-grained-scoping options have been reviewed against this project's actual usage yet, which is a real gap, not an oversight glossed over — track it as a follow-up rather than asserting a scoping decision that hasn't actually been made.

## Rotation

Rotate every key on a fixed schedule — 90 days is this project's default, matching the same window `docs/security/SECURITY.md` already uses for episodic-memory auto-expiry, so there's one cadence to remember rather than several — and immediately, out of cycle, the moment any of the leak indicators below fire. Rotating means generating a new key in the provider's console, updating the local `.env`, confirming the application picks up the new value, and only then revoking the old key — revoking first and generating second creates a working-service gap for no safety benefit, since the old key was never exposed by the act of rotating on schedule.

## What counts as a leak, and what to do about one

A leak is any case where a real key value left `.env` and reached somewhere this project doesn't control — a committed file, a pasted chat log, a public gist, a CI log that echoed an environment variable, a screenshot. The moment one is suspected:

1. **Revoke the key in the provider's console first**, before doing anything else — a revoked key stops being useful to whoever has it, immediately, and every other step can happen after that's done.
2. **Generate a replacement key and update `.env`.**
3. **Check the provider's usage dashboard for the leaked key's activity in the window since it was last known-safe** — unexpected usage is the concrete signal that the leak was actually exploited, not just theoretically possible.
4. **If the leak was a git commit**, the key is compromised even if the commit is later removed or force-pushed over — git history is recoverable by anyone who already cloned or fetched it, so step 1 (revoke) is what actually neutralizes this, not any amount of history rewriting.

## The mechanical layer: a pre-commit hook, not just a rule

A written rule against committing secrets only works if everyone remembers it, every time — the same "automate it, don't hand-move it" reasoning this project already applies to GitHub project boards (`docs/governance` skill-routing) applies here too. This repository ships a real pre-commit hook at `.githooks/pre-commit` that blocks a commit outright if it stages a file named exactly `.env`, or if any staged file's content matches a known API-key shape (`sk-ant-*` for Anthropic, `AIza*` for Google/Gemini, and a generic high-entropy long-token heuristic as a catch-all). It is a real, working script — not a description of one — and running it costs a fraction of a second per commit.

**It only runs if it's wired in.** Git doesn't activate hooks under `.githooks/` automatically; run this once per clone:

```bash
git config core.hooksPath .githooks
```

This is a project-level `.gitignore`-adjacent gap worth naming honestly: a hook that isn't wired in provides exactly zero protection despite existing in the repository, which is precisely the "a rule that isn't enforced isn't really a rule" problem this section exists to close. Confirm it's active with `git config --get core.hooksPath` — it should print `.githooks`.

The hook is a defense-in-depth layer, not a replacement for `.gitignore` — `.gitignore` stops `.env` from ever being staged in the first place, and the hook catches the cases `.gitignore` can't: a key pasted directly into a tracked file, a key committed before this protocol existed, or a `.gitignore` entry that gets accidentally edited or removed later. Neither layer alone is the protocol; both together are.
