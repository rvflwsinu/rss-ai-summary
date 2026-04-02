# AGENTS.md

## Scope

- This file applies to the entire repository.
- No prior `AGENTS.md` was present during this scan.
- No `.cursorrules` file was found.
- No `.cursor/rules/` directory was found.
- No `.github/copilot-instructions.md` file was found.
- Treat this file as the canonical instruction set for agentic coding work here.
- This is a plain Python script repository, not a packaged library.

## Project Layout

- `src/main.py` is the CLI entrypoint.
- `src/rss_digest.py` contains nearly all runtime logic.
- `subscriptions.opml` is the feed configuration input.
- `data/state.json` stores persistent dedupe state.
- `data/latest.json` is the latest generated report.
- `site/index.html` and `site/.nojekyll` are generated site artifacts.
- `requirements.txt` is the only committed dependency manifest.
- `tests/` exists but is currently empty.
- No `pyproject.toml`, `pytest.ini`, `tox.ini`, `ruff.toml`, `.flake8`, `mypy.ini`, or `Makefile` were found.

## Setup

- Run commands from the repository root.
- Use `python3`, not `python`, unless the environment clearly aliases it.
- The code uses modern typing syntax such as `str | None` and `list[str]`.
- Assume Python 3.10+ at minimum.
- Recommended setup: `python3 -m venv .venv`
- Install dependencies with `.venv/bin/pip install -r requirements.txt`.
- `src/main.py` imports `rss_digest` directly, so run it as `python3 src/main.py`.

## Commands

- CLI smoke test: `python3 src/main.py --help`
- Local generation without Gemini: `python3 src/main.py --mock-summary`
- Local generation with Gemini: `GEMINI_API_KEY=your_key_here python3 src/main.py`
- Syntax check: `python3 -m compileall src`
- Test discovery: `python3 -m unittest discover -s tests -v`
- Single test status: none exists today because `tests/` is empty.
- If you add `unittest` tests, run one with `python3 -m unittest tests.test_rss_digest.TestSomething.test_case_name`.
- There is no configured formatter, linter, or static type checker in the repository.
- Do not claim `ruff`, `black`, `pytest`, `mypy`, or `pyright` are standard unless you add and document them.

## Generated Artifacts

- Avoid hand-editing `data/state.json`, `data/latest.json`, `site/index.html`, or `site/.nojekyll` unless the task is specifically about generated output.
- Prefer changing source code and regenerating outputs when needed.
- Running the full script updates tracked artifacts.
- Only regenerate outputs when the task actually requires it.
- Preserve persisted state compatibility unless the user approves a migration.

## Import Style

- Keep `from __future__ import annotations` first when present.
- Group standard-library imports together.
- Group third-party imports after standard-library imports.
- Separate import groups with a single blank line.
- Prefer explicit imports over wildcard imports.
- Remove unused imports.

## Formatting Style

- Use 4-space indentation.
- Match the existing Black-like formatting style.
- Use double quotes consistently.
- Break long calls and literals over multiple lines.
- Use trailing commas in multiline structures when they improve stability.
- Use `textwrap.dedent` for large prompt strings or HTML templates.
- Prefer readable f-strings over manual concatenation.
- Keep formatting changes narrow and local.

## Types And Data

- Add type hints to new functions and return values.
- Use built-in generic syntax like `list[str]`, `dict[str, Any]`, and `tuple[str, str]`.
- Prefer `str | None` over `Optional[str]`.
- Use `Path` for filesystem paths.
- Use `@dataclass` for small structured records such as feed metadata.
- Do not introduce class hierarchies when a dataclass or dict is enough.
- Keep public data shapes stable if downstream artifacts depend on them.
- Do not replace working dict-based report structures casually.
- Keep data structures simple unless a broader refactor is requested.

## Naming And Functions

- Use `snake_case` for functions, variables, and helpers.
- Use `UPPER_SNAKE_CASE` for module-level constants.
- Use `PascalCase` for dataclasses and classes.
- Prefer descriptive verb phrases for functions.
- Prefer descriptive noun phrases for structured data.
- `tldr` is an established output field; keep it where the data contract already uses it.
- Prefer small, single-purpose helpers.
- Keep side effects near CLI, HTTP, and file I/O boundaries.
- Reuse existing helpers before adding near-duplicates.

## Error Handling

- Raise specific exceptions for invalid configuration or malformed inputs.
- Use `raise ... from exc` when translating lower-level errors.
- Validate environment variables before use.
- Fail fast for missing required files such as the OPML input.
- Call `response.raise_for_status()` on HTTP responses.
- Catch broad exceptions only at integration boundaries.
- Current broad-catch boundaries include feed fetches, article extraction, and Gemini summarization.
- Preserve fallback behavior unless the user asks to change it.
- Do not silently swallow errors that should surface to the user or report output.

## HTTP And External Services

- Reuse a shared `httpx.Client` when touching feed-fetching code.
- Keep explicit timeouts and the custom `USER_AGENT` unless there is a reason to change them.
- Keep retries modest.
- Treat feed data, article pages, and LLM output as untrusted input.
- Normalize external text before reuse.
- Preserve the strict Gemini JSON contract in `call_gemini_json()`.
- Preserve the `--mock-summary` flow for local testing.
- Do not add new services or SDKs for small changes.

## HTML And Output

- Escape feed-provided text before interpolating it into HTML.
- Keep the generated HTML self-contained.
- Preserve mobile-friendly behavior when changing layout or CSS.
- Use UTF-8 for file reads and writes.
- Keep JSON output pretty-printed with `indent=2` and a trailing newline.
- Continue using `Path.read_text()` and `Path.write_text()` patterns.
- Create parent directories with `mkdir(parents=True, exist_ok=True)` when needed.
- Avoid changing report keys unless the task explicitly requires a schema change.
- Maintain the current static-site approach unless the user requests a different frontend.

## Testing

- Add tests under `tests/` for non-trivial logic changes.
- Prefer `unittest` unless the user explicitly wants another framework.
- Focus tests on pure helpers first.
- Mock network calls instead of hitting real feeds.
- Mock Gemini responses instead of calling the live API.
- Keep fixtures minimal and local to the test file.
- Ensure new tests run with `python3 -m unittest discover -s tests -v`.
- If you add just one test, verify it with a dotted-path invocation too.
- If you do not add tests, say so explicitly in the final response.
- If you only ran `--help` or `compileall`, report that accurately.

## Change Strategy

- Read the relevant part of `src/rss_digest.py` before editing.
- Prefer the smallest correct change over a broad refactor.
- Do not assume hidden tooling or CI exists.
- Do not add packaging files or developer tooling unless the task calls for it.
- Avoid changing CLI flags, environment variable names, or output locations without a reason.
- Keep dependencies limited to what is already in `requirements.txt` unless asked.
- Keep import ordering and nearby formatting stable.
- Keep source changes in source files, not only in generated artifacts.
- If future Cursor or Copilot rule files appear, merge them with this file rather than ignoring them.
- When unsure, follow the patterns already established in `src/rss_digest.py`.

## Done Criteria

- Changes are minimal and repository-specific.
- New behavior is implemented in source code, not only in generated output.
- Commands mentioned in summaries are real commands for this repo.
- Missing tooling is described honestly instead of guessed.
- Meaningful logic changes include tests when practical.
- Verification steps are reported accurately.
- Generated files are only changed intentionally.
- Final responses should clearly say what changed and what was verified.
