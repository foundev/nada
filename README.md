# nada

A minimal, stdlib-only coding agent (`nada.py`) driven by a model via
OpenRouter. It exposes four tools to the model: `read_file`, `write_file`,
`edit_file`, and `bash`.

## Usage

```sh
export OPENROUTER_API_KEY=...        # MODEL defaults to meta/muse-ember
python3 nada.py                       # interactive REPL
python3 nada.py "add a README"        # one-shot task
python3 nada.py --help
```

REPL commands: `/quit`, `/clear`, `/mode [MODE]`, `/permissions [MODE|clear]`,
`/effort [LEVEL|clear]`.

## Permissions

Mutating and shell tools are gated by permission prompts. `read_file` and
provably read-only `bash` commands never prompt, in any mode.

### Modes

Claude-Code-style permission modes. The current mode is shown in the REPL
prompt, e.g. `[accept-edits] > `, at startup, and by `/mode` / `/permissions`.

| Mode | Behaviour |
|------|-----------|
| `default` | Prompt before `write_file`, `edit_file`, and non-read-only `bash` |
| `accept-edits` | `write_file`/`edit_file` auto-approved; non-read-only `bash` still prompts |
| `plan` | Read-only only; edits and non-read-only `bash` are denied without prompting |
| `auto` | Codex-style auto-review ("approve for me"): gated actions are approved/denied by a separate reviewer model (see below) |
| `yolo` | Allow everything, never prompt, never review (same as `--yes` / bypass permissions) |

Aliases (all accepted anywhere a mode is): `ask` → `default`,
`approve-edits` / `acceptEdits` → `accept-edits`, `deny` / `dont-ask` → `plan`,
`approve-for-me` / `auto-review` / `guardian` → `auto`,
`bypass-permissions` → `yolo`.

Set the mode with `--permission-mode MODE` (or `--mode`), `--accept-edits`,
`--plan`, `--yes`/`-y` (yolo), the `AGENT_PERMISSION_MODE` environment
variable, or in the REPL with `/mode MODE` / `/permissions MODE` (CLI flags
override the env var; `--permission-mode` overrides `--accept-edits`/`--plan`/
`--yes`). `/permissions` with no argument also shows the session allowlist;
`/permissions clear` forgets it.

### Auto-review (`auto` mode)

Like Codex's `approvals_reviewer = "auto_review"` / "Approve for me": every
*eligible* gated action — i.e. one that would otherwise prompt you — is sent
to a separate reviewer model instead of to you. Actions already allowed
(`read_file`, provably read-only `bash`, session allowlist) run without
review. The reviewer sees the exact action plus recent conversation context
and must reply `{"decision": "approve"|"deny", "rationale": ...}` under a
guardian policy: deny exfiltration, credential probing, persistent security
weakening, and destructive/irreversible actions; approve ordinary reversible
in-workspace work. Reviewer errors and unparsable verdicts **fail closed**
(deny). This costs one extra model call per gated action.

The reviewer uses `AGENT_REVIEW_MODEL` (default `openai/gpt-6-luna`); e.g.
`AGENT_REVIEW_MODEL=openai/gpt-5.6-luna` to override it, if available on
OpenRouter. `yolo` is the old behaviour of `auto`: no prompt, no review.

### Switching modes with Ctrl+Tab

At the REPL prompt, press **Ctrl+Tab** (or Shift+Tab) to cycle
`default → accept-edits → plan → auto → yolo → default`; Ctrl+Shift+Tab cycles
backwards. The prompt redraws with the new mode immediately, and any text
you've typed is kept.

Terminals don't agree on a Ctrl+Tab encoding, so several are recognised
(kitty/xterm `ESC[27;5;9~`-style sequences and Shift+Tab `ESC[Z`). Some
terminals can't distinguish Ctrl+Tab from plain Tab — those send a plain Tab,
which also cycles (the REPL has no tab-completion to give up). Without a TTY
there's no cycling; use `/mode`.

### Prompting

For each gated call you are shown the tool, its arguments (a preview of the
file contents / edit, or the shell command), and asked:

- `y` — allow once
- `a` — always allow *this exact command or file path* for the rest of the
  session (a different command/path prompts again; `/permissions clear`
  forgets these)
- `n` — deny; the model receives a `PermissionDenied` result and can adapt

EOF/Ctrl-C at the prompt denies the call.

### Read-only auto-allow

`bash` commands are auto-allowed only if they are a single, simple,
provably read-only command: `ls`, `rg`, `grep`, `cat`, `head`, `tail`,
`wc`, `pwd`, `diff`, `stat`, `file`, `find` (no `-exec`/`-delete`/…), and
read-only `git` subcommands (`status`, `log`, `diff`, `show`, `blame`,
`describe`, `rev-parse`, `ls-files`). Anything containing shell
metacharacters — `;`, `&&`, `|`, `>`, `$`, backticks, newlines — or
anything else (e.g. `rm`, `make`, `pytest`, `git add`, `./script.sh`,
`sudo …`) prompts. This check is intentionally conservative: a false
positive just costs a prompt.

### Non-interactive use

In `default` mode with no TTY on stdin, gated calls are **denied** (fail
closed) rather than hanging on a prompt. For scripted runs, pass `--yes`
(yolo), `--accept-edits`, or set `AGENT_PERMISSION_MODE=yolo`. `auto` also
works non-interactively — gated actions go to the reviewer model instead of
a prompt (and are denied if the reviewer is unavailable).

## Model effort level

The model's reasoning effort is settable, and the available levels are
*discovered* from OpenRouter rather than hard-coded: at startup nada fetches
`GET https://openrouter.ai/api/v1/model/{MODEL}` (no API key needed, cached
per model) and reads `reasoning.supported_efforts` (highest first),
`reasoning.default_effort`, and `reasoning.mandatory` from the model's own
metadata. Which levels exist depends on the model — e.g. the default
`meta/muse-ember` accepts `max|xhigh|high|medium|low|minimal` (mandatory
reasoning, default `max`, so `none` is rejected), while `openai/gpt-5.2`
accepts `xhigh|high|medium|low|none`. Models whose `reasoning` has no
`supported_efforts` expose no effort selection, and setting one is rejected.

A level is sent on chat completions as `"reasoning": {"effort": LEVEL}`.
Unset (the default) sends nothing, so the model's own default applies; `none`
(where the model offers it) disables reasoning. It applies to the main model
only — a different `auto`-mode reviewer model uses its own default, so an
unsupported level can't break reviewer calls. Discovery is best-effort: if
the metadata request fails (e.g. offline) nada warns and continues, and a
level is then validated only against the gateway's full set
(`max|xhigh|high|medium|low|minimal|none`), leaving OpenRouter to judge.

Set it with `--effort LEVEL` (or `--reasoning-effort LEVEL`), the
`AGENT_EFFORT` environment variable (aliases `MODEL_EFFORT` /
`REASONING_EFFORT` / `EFFORT`), or in the REPL with `/effort LEVEL`; `/effort` alone
shows the levels available for the current model, and `/effort clear` (or
`default`/`reset`) unsets it back to the model default. Aliases: `off` →
`none`, `x-high`/`extra-high` → `xhigh`. An invalid level for the model exits
with an error listing the available ones.

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | — | required |
| `MODEL` | `meta/muse-ember` | model id |
| `AGENT_REVIEW_MODEL` | `openai/gpt-6-luna` | reviewer model id for `auto` mode |
| `AGENT_PERMISSION_MODE` | `default` | `default`\|`accept-edits`\|`plan`\|`auto`\|`yolo` (aliases: `ask`, `deny`, `approve-for-me`, …) |
| `AGENT_EFFORT` | model default | reasoning effort level for `MODEL`, discovered per model (see above; aliases `MODEL_EFFORT`, `REASONING_EFFORT`, `EFFORT`) |
| `AGENT_MAX_TURNS` | `2000000` | max tool-calling turns per task |
| `AGENT_MAX_RETRIES` | `5` | retries on HTTP 429 (rate limit) before giving up |
| `AGENT_RETRY_BASE_DELAY` | `1` | seconds; first 429 wait (doubled every retry) |
| `AGENT_RETRY_MAX_DELAY` | `60` | cap in seconds on a single backoff wait (`Retry-After` can exceed it) |
| `AGENT_RETRY_BACKOFF` | `2` | multiplier applied to the wait after each 429 |

## Rate limits (429)

HTTP 429 responses from OpenRouter are retried with exponential backoff
(`1s, 2s, 4s, …`, capped, with jitter) up to `AGENT_MAX_RETRIES` times,
honouring a server `Retry-After` header as a minimum wait. This applies to
both the main model calls and the `auto`-mode reviewer calls, since both go
through `call_model`. Any other HTTP error raises immediately, as does a
429 that persists past the last retry.

## Tests

```sh
python3 -m unittest -v
```
