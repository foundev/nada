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

REPL commands: `/quit`, `/clear`, `/permissions [ask|auto|deny|clear]`.

## Permissions

Mutating and shell tools are gated by permission prompts. `read_file` and
provably read-only `bash` commands never prompt.

### Modes

| Mode   | Behaviour |
|--------|-----------|
| `ask`  | Prompt before `write_file`, `edit_file`, and non-read-only `bash` (default) |
| `auto` | Allow everything (same as `--yes`) |
| `deny` | Allow only read-only tools; deny everything else |

Set the mode with `--permission-mode ask|auto|deny`, `--yes`/`-y`
(auto-approve), the `AGENT_PERMISSION_MODE` environment variable, or in the
REPL with `/permissions auto|ask|deny` (CLI flags override the env var).
The current mode is shown at startup and by `/permissions`.

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

In `ask` mode with no TTY on stdin, gated calls are **denied** (fail
closed) rather than hanging on a prompt. For scripted runs, pass `--yes`
or set `AGENT_PERMISSION_MODE=auto`.

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | — | required |
| `MODEL` | `meta/muse-ember` | model id |
| `AGENT_PERMISSION_MODE` | `ask` | `ask`\|`auto`\|`deny` |
| `AGENT_MAX_TURNS` | `2000000` | max tool-calling turns per task |

## Tests

```sh
python3 -m unittest -v
```
