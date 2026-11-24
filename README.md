# nada

A minimal, stdlib-only coding agent (`nada.py`) driven by a model via
OpenRouter or an automatically detected local llama.app server. It exposes four tools to the model: `read_file`, `write_file`,
`edit_file`, and `bash`.

## Usage

```sh
export OPENROUTER_API_KEY=...        # MODEL defaults to meta/muse-ember
python3 nada.py                       # interactive REPL
python3 nada.py "add a README"        # one-shot task
python3 nada.py --help
```

If a llama.app server is running at `http://127.0.0.1:9931`, nada detects it
via `/v1/models`, uses its first advertised model, and sends requests to its
OpenAI-compatible `/v1/chat/completions` endpoint. No `OPENROUTER_API_KEY` is
needed in that case. Set `MODEL` or pass `--model` to select a different
served model; set `LLAMA_APP_URL` to use a non-default address, and
`LLAMA_APP_API_KEY` if the server was started with `--api-key`.

REPL commands: `/help`, `/status`, `/verbose [on|off]`, `/quit`, `/clear`,
`/mode [MODE]`, `/permissions [MODE|clear]`, `/effort [LEVEL|clear]`,
`/model [MODEL_ID]`,
`/session [save [TITLE]|list|resume ID|new|delete ID|path]`
(aliases: `/sessions`, `/resume ID`).

## Terminal UI

The REPL borrows Codex's look (still line-based and stdlib-only — no
alternate screen):

- A slim boxed header at startup with only what the statusline doesn't show:
  the `nada` name, the full working-directory path (the bar shows just its
  basename), any `AGENTS.md` files loaded, and a help hint — plus an effort
  summary line below it (the available effort levels, not just the current
  one the bar shows).
- A bare prompt — `> ` (bold on a tty). The permission mode lives in the
  statusline below, not in the prompt; only with no statusline (piped /
  dumb terminal) does the prompt carry it as `[mode] > `, with the mode in
  its own colour (`plan` yellow, `yolo` red, `auto` cyan,
  `accept-edits` green). Colour is tty-only: piped output and one-shot
  runs stay plain text, and `$NO_COLOR` (or `--no-color`) disables it
  everywhere.
- A statusline bar on the line under the prompt, holding all live state —
  `meta/muse-ember · nada · default · effort: max · session: s1 · 3 turns · tokens: 1,234 in · 567 out`
  — the model, working directory, permission mode (in its colour, its only
  appearance on a tty), effort, session, turn count, and the last turn's
  token usage. It spans the full terminal width (low-priority segments are
  dropped first on a
  narrow terminal), refreshes instantly when Ctrl+Tab cycles the mode, and
  is drawn only while a prompt is open — submitting clears it and the
  output below takes its line. (Line-based, no alternate screen: that's as
  pinned as a bar gets without a full TUI.) Without a tty there is no bar;
  in cooked-mode fallback it's printed above the prompt instead.
- An animated `⠋ Working (2s)` spinner while the model or a long tool call
  runs (tty only). Press **Esc** (or Ctrl-C) while a turn is running to
  interrupt it and return to the prompt — the model/tool call is aborted
  (a running `bash` command is killed) and any partial tool calls are
  discarded, so the next turn starts clean.
- Collapsed tool lines — `✓ Ran python3 -m unittest -v · 1.2s`,
  `✓ Edited nada.py`, `✓ Read nada.py · 1425 lines`, `✗ Ran make · exit 1` —
  with command output indented and dimmed underneath, capped at 12 lines
  (`… N more lines`). Reads/edits show no body, and neither do raw argument
  dumps. `/verbose on` (or `--verbose` / `$NADA_VERBOSE=1`) shows full tool
  output instead. A turn that uses tokens ends with a dim
  `tokens: 1,234 in · 567 out` footer.
- Approval prompts show the action as a `$ command` line or a `+`/`-` diff,
  with numbered options (`1`/`2`/`3`, same as `y`/`a`/`n`).

`/help` lists all commands; `/status` shows the model, root, permissions,
effort, verbose setting, and session.

## AGENTS.md

nada follows the [AGENTS.md](https://agents.md) spec, and nothing beyond it:
`AGENTS.md` is plain Markdown with no required fields/headings/frontmatter,
and its text is included verbatim in the system prompt.

- Place `AGENTS.md` at the root of the repository (nada's working
  directory). The exact filename is `AGENTS.md` — no other name or location
  (e.g. no global/home file, no `AGENT.md`/`CLAUDE.md` fallback) is consulted.
- For a large monorepo, place a nested `AGENTS.md` inside each
  subproject/subdirectory. nada loads the root file and every nested file
  under the root, ordered from the root to the deepest nested file, each
  labelled with its path (at startup the loaded files are listed).
- Precedence, per the spec: where `AGENTS.md` instructions conflict, the
  closest `AGENTS.md` to the file being edited wins; explicit user
  instructions in the chat override `AGENTS.md`.
- The system prompt is rebuilt for new/restored/cleared sessions, so
  editing an `AGENTS.md` (living documentation) is picked up on the next
  such session. An unreadable `AGENTS.md` is skipped with a warning.

## Permissions

Mutating and shell tools are gated by permission prompts. `read_file` and
provably read-only `bash` commands never prompt, in any mode.

### Modes

Claude-Code-style permission modes. The current mode is shown in the REPL
statusline bar under the prompt (in the `[accept-edits] > ` prompt itself
only when no statusline is drawn, e.g. piped), and by `/mode` / `/permissions`.

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
`/permissions clear` forgets it. A mode selected in nada is saved for the
next run (see Model, effort, and permission-mode persistence below).

### Auto-review (`auto` mode)

Like Codex's `approvals_reviewer = "auto_review"` / "Approve for me": every
*eligible* gated action — i.e. one that would otherwise prompt you — is sent
to a separate reviewer model instead of to you. Actions already allowed
(`read_file`, provably read-only `bash`, session allowlist) run without
review. The reviewer sees the **exact action, untruncated**, plus the **full
conversation back-and-forth, untruncated**, in labelled sections: every user
request is always retained verbatim as the (only) trusted source of
authorization, while assistant/tool content is untrusted evidence. It scores
`risk_level` and `user_authorization` and derives the decision by thresholds
(low/medium risk → approve; high → approve only with at least medium user
authorization; critical → deny), denying exfiltration, credential probing,
persistent security weakening, and destructive/irreversible actions under any
score. A user re-approval of a previously denied action counts as high
authorization. The verdict is requested as strict JSON. A first denial is
**resampled once** before it stands, and a reviewer error is retried once;
anything still failing **fails closed** (deny). The denial reason (or
reviewer error) is included in the `PermissionDenied` result the model
receives, so it can adapt. This costs one extra model call per gated action
(two if the first verdict is a denial or errors).

The reviewer uses `AGENT_REVIEW_MODEL` (default `openai/gpt-6-luna`); e.g.
`AGENT_REVIEW_MODEL=openai/gpt-5.6-luna` to override it, if available on
OpenRouter. `yolo` is the old behaviour of `auto`: no prompt, no review.

### Switching modes with Ctrl+Tab

At the REPL prompt, press **Ctrl+Tab** (or Shift+Tab) to cycle
`default → accept-edits → plan → auto → yolo → default`; Ctrl+Shift+Tab cycles
backwards. The statusline redraws with the new mode immediately, any
text you've typed is kept, and the cycled-to mode is saved for the next
run.

Terminals don't agree on a Ctrl+Tab encoding, so several are recognised
(kitty/xterm `ESC[27;5;9~`-style sequences and Shift+Tab `ESC[Z`). Some
terminals can't distinguish Ctrl+Tab from plain Tab — those send a plain Tab,
which also cycles (the REPL has no tab-completion to give up). Without a TTY
there's no cycling; use `/mode`.

### Prompting

For each gated call you are shown the action (a `$ command` line or a
`+`/`-` diff of the edit), and asked — the numbered options are the same
answers as the letters:

- `1` / `y` — allow once
- `2` / `a` — always allow *this exact command or file path* for the rest of
  the session (a different command/path prompts again; `/permissions clear`
  forgets these)
- `3` / `n` — deny; the model receives a `PermissionDenied` result and can
  adapt

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

When using an automatically detected llama.app server, this OpenRouter-only
metadata is unavailable, so reasoning effort selection is disabled.

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

The model itself is selected with `--model MODEL_ID`, the `MODEL` environment
variable, or in the REPL with `/model MODEL_ID` (`/model` alone shows the
current one). Switching model clears the effort if the new model does not
offer it.

## Model, effort, and permission-mode persistence

The last model, effort, and permission mode selected *in nada* are saved
to `~/.nada/settings.json` and reloaded on the next run, so a fresh
session starts with your selection instead of the built-in defaults:

```json
{"version": 1, "model": "openai/gpt-5.2", "effort": "high",
 "permission_mode": "accept-edits", "updated_at": "..."}
```

- Saved on: `--model` / `--effort` / `--permission-mode` (and
  `--yes` / `--accept-edits` / `--plan`) on the command line, `/model` /
  `/effort` / `/mode` (including `/effort clear`, which saves the model
  default) in the REPL, Ctrl+Tab cycling at the prompt, and — for
  model/effort only — when a session is resumed.
- Precedence at startup: command line > environment variables (`MODEL` /
  the effort vars / `AGENT_PERMISSION_MODE`) > saved settings > built-in
  default (`meta/muse-ember`, model default effort, `default` mode).
  Env vars are *not* saved — they already override every run they're set
  in.
- A saved effort the current model does not support (e.g. after switching
  models) is dropped with a warning, not fatal; an invalid saved mode is
  dropped likewise; a corrupt settings file is ignored with a warning.
  Writes are atomic (temp file + rename), like sessions. Override the
  location with `NADA_SETTINGS_FILE`. `--no-autosave` only affects
  sessions, not settings.
- Note the asymmetry with sessions: a *session* file never restores a
  permission mode (loading a file must not silently escalate), but the
  settings file is your own recorded preference, so a saved mode —
  including `yolo` — does apply on the next run. It is always visible in
  the statusline, and `--permission-mode`/the env var still override it
  for that run.

## Sessions

Conversations are persisted as sessions and survive quitting. A session is
one JSON file (`<id>.json`) under `~/.nada/sessions` (override with
`--session-dir DIR` or `AGENT_SESSION_DIR`), storing the messages verbatim
(so `tool_calls` resume correctly), the REPL history, and the model, effort,
and permission mode in force. The title is derived from the first user
message unless you name the session with `/session save TITLE`. Writes are
atomic (temp file + rename), and a session interrupted mid-turn is trimmed
back to the last complete tool call, so resuming never sends the API a
dangling `tool_calls` message.

Sessions are **autosaved** after every REPL turn and after one-shot runs
(`--no-autosave` disables this; `/session save` still works), and once more
on exit. A session with no turns yet is never written.

```sh
python3 nada.py                     # new session (id in the statusline)
python3 nada.py --continue          # resume the most recently saved session
python3 nada.py --session ID        # resume a specific session ('latest' also works)
python3 nada.py --session ID "follow-up task"   # resume + one-shot task, then save
```

In the REPL: `/session` shows the current session, `/session list` lists
saved sessions (newest first), `/session resume ID` (or `/resume ID`) loads
one, `/session new` starts a fresh one, and `/session delete ID` removes
one (deleting the live session starts a new one, so autosave can't
resurrect it). Resuming regenerates the system prompt for the current
working directory and reports it if the session was created elsewhere;
the saved model and effort are restored (unless `--model` / `--effort`
was given), but the permission mode and session allowlist are deliberately
**not** restored from the session (a saved *settings* mode still applies —
see persistence above) —
loading a file must not silently escalate permissions, so re-apply them
with `/mode` / the usual prompts. Session ids are simple file stems;
anything else (`..`, `/`) is rejected, unknown ids are an error, and
corrupt session files are skipped with a warning when listing.

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | — | required unless llama.app is detected |
| `LLAMA_APP_URL` | `http://127.0.0.1:9931` | local llama.app server address; automatically used when its `/v1/models` endpoint is available |
| `LLAMA_APP_API_KEY` | — | optional Bearer token for a llama.app server started with `--api-key` |
| `MODEL` | saved model, else `meta/muse-ember` | model id (`--model` overrides; see Model and effort persistence) |
| `AGENT_REVIEW_MODEL` | `openai/gpt-6-luna` | reviewer model id for `auto` mode |
| `AGENT_PERMISSION_MODE` | saved mode, else `default` | `default`\|`accept-edits`\|`plan`\|`auto`\|`yolo` (aliases: `ask`, `deny`, `approve-for-me`, …; CLI flags override) |
| `AGENT_EFFORT` | saved effort, else model default | reasoning effort level for `MODEL`, discovered per model (see above; aliases `MODEL_EFFORT`, `REASONING_EFFORT`, `EFFORT`) |
| `NADA_SETTINGS_FILE` | `~/.nada/settings.json` | file the selected model/effort/permission mode are saved to |
| `AGENT_SESSION_DIR` | `~/.nada/sessions` | directory for saved sessions (alias `NADA_SESSION_DIR`; `--session-dir` overrides) |
| `AGENT_MAX_TURNS` | `2000000` | max tool-calling turns per task |
| `AGENT_MAX_RETRIES` | `5` | retries on HTTP 429 (rate limit) before giving up |
| `AGENT_RETRY_BASE_DELAY` | `1` | seconds; first 429 wait (doubled every retry) |
| `AGENT_RETRY_MAX_DELAY` | `60` | cap in seconds on a single backoff wait (`Retry-After` can exceed it) |
| `AGENT_RETRY_BACKOFF` | `2` | multiplier applied to the wait after each 429 |
| `NADA_VERBOSE` | off | show full tool output instead of the collapsed view (`--verbose` / `/verbose on` overrides) |
| `NO_COLOR` | — | disable coloured output (or pass `--no-color`) |

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
