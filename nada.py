#!/usr/bin/env python3
import argparse, json, os, random, re, select, shlex, subprocess, sys, time, urllib.error, urllib.parse, urllib.request, uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

try:
    import termios, tty
except ImportError:  # non-Unix: REPL falls back to plain input()
    termios = tty = None

ROOT = Path.cwd().resolve()
MODEL = os.environ.get("MODEL", "meta/muse-ember")
# Reviewer for `auto` (approve-for-me) mode; defaults to openai/gpt-6-luna.
# Override e.g. AGENT_REVIEW_MODEL=openai/gpt-5.6-luna (if available on OpenRouter).
REVIEW_MODEL = os.environ.get("AGENT_REVIEW_MODEL") or os.environ.get("REVIEW_MODEL") or "openai/gpt-6-luna"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "2000000"))
# 429 (rate limit) retries in call_model: exponential backoff between tries.
MAX_RETRIES = int(os.environ.get("AGENT_MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.environ.get("AGENT_RETRY_BASE_DELAY", "1"))   # seconds
RETRY_MAX_DELAY = float(os.environ.get("AGENT_RETRY_MAX_DELAY", "60"))    # seconds
RETRY_BACKOFF = float(os.environ.get("AGENT_RETRY_BACKOFF", "2"))         # multiplier per retry
API_URL = "https://openrouter.ai/api/v1/chat/completions"
# Per-model metadata (singular "model"): reasoning.supported_efforts etc.
MODEL_INFO_URL = "https://openrouter.ai/api/v1/model"
BASH = "/bin/bash" if Path("/bin/bash").exists() else None

# --- Model effort level (reasoning effort) ----------------------------------
# All effort values the OpenRouter gateway understands, highest first. A given
# model usually accepts only a subset -- discover it from the model itself
# (fetch_model_reasoning / discover_efforts) rather than hard-coding it here.
# None = unset: send no `reasoning` param and let the model default apply.
EFFORT_LEVELS = ("max", "xhigh", "high", "medium", "low", "minimal", "none")
_EFFORT_ALIASES = {"off": "none", "disabled": "none", "disable": "none",
                   "x-high": "xhigh", "x_high": "xhigh",
                   "extra-high": "xhigh", "extra_high": "xhigh"}
_EFFORT_CLEAR = {"default", "clear", "reset", "unset", "model-default", "model_default"}

def normalize_effort(effort):
    """Canonical effort level, or None for unset/clear/default/empty."""
    if effort is None:
        return None
    key = str(effort).strip().lower()
    if not key or key in _EFFORT_CLEAR:
        return None
    return _EFFORT_ALIASES.get(key, key)

EFFORT = normalize_effort(os.environ.get("AGENT_EFFORT") or os.environ.get("MODEL_EFFORT")
                          or os.environ.get("REASONING_EFFORT") or os.environ.get("EFFORT"))
_MODEL_REASONING_CACHE = {}  # model id -> parse_model_reasoning() dict

def validate_effort(effort, supported=None):
    """Normalized effort (or None), raising ValueError if it is not a real
    level, or -- when `supported` (the discovered per-model list) is given --
    if the model does not accept it. `supported=None` means unknown: any
    gateway level passes and OpenRouter judges."""
    level = normalize_effort(effort)
    if level is None:
        return None
    if level not in EFFORT_LEVELS:
        raise ValueError(f"invalid effort level: {effort!r} "
                         f"(want one of {', '.join(EFFORT_LEVELS)})")
    if supported is not None and level not in supported:
        avail = ", ".join(supported) if supported else "none (model does not expose effort selection)"
        raise ValueError(f"effort {level!r} not supported by {MODEL} (available: {avail})")
    return level

def set_effort(effort, supported=None):
    """Validate and set the session effort; unchanged on failure."""
    global EFFORT
    EFFORT = validate_effort(effort, supported)
    return EFFORT

def parse_model_reasoning(model_data):
    """Reasoning info from an OpenRouter model object: supported_efforts
    (highest first; [] when the model exposes no effort selection),
    default_effort, mandatory, default_enabled. A present-but-null
    supported_efforts means all gateway levels are accepted."""
    reasoning = (model_data or {}).get("reasoning") or {}
    if "supported_efforts" not in reasoning:
        supported = []
    elif reasoning["supported_efforts"] is None:
        supported = list(EFFORT_LEVELS)
    else:
        supported = [str(e).strip().lower() for e in reasoning["supported_efforts"] if str(e).strip()]
    default = reasoning.get("default_effort")
    return {"supported_efforts": supported,
            "default_effort": str(default).strip().lower() if default else None,
            "mandatory": bool(reasoning.get("mandatory")),
            "default_enabled": reasoning.get("default_enabled")}

def fetch_model_reasoning(model=None, *, timeout=15):
    """Reasoning info for `model` (default MODEL), discovered live from
    GET /api/v1/model/{model} and cached per model id. Raises on HTTP /
    network errors so callers can fail soft. No API key needed."""
    model = model or MODEL
    if model in _MODEL_REASONING_CACHE:
        return _MODEL_REASONING_CACHE[model]
    url = f"{MODEL_INFO_URL}/{urllib.parse.quote(model, safe='/:')}"
    req = urllib.request.Request(url, headers={"X-Title": "stdlib-agent"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    info = parse_model_reasoning(data.get("data") or data)
    _MODEL_REASONING_CACHE[model] = info
    return info

def discover_efforts(model=None, *, timeout=15):
    """Available effort levels for `model`, highest first ([] if none)."""
    return list(fetch_model_reasoning(model, timeout=timeout)["supported_efforts"])

def _effort_summary(info):
    """One-line current + available effort summary; `info` is a
    parse_model_reasoning() dict, or None if discovery failed."""
    current = EFFORT or "(model default)"
    if info is None:
        avail = "unknown -- could not query OpenRouter"
    elif not info["supported_efforts"]:
        avail = "none -- model does not expose effort selection"
    else:
        avail = ", ".join(info["supported_efforts"])
        if info.get("default_effort"):
            avail += f" (model default: {info['default_effort']})"
    return f"Effort: {current} for {MODEL} (available: {avail})"

SYSTEM = f"""You are a coding agent in {ROOT}.
Explore with bash: `rg -n PATTERN .` (or `grep -rn PATTERN .`), `find . -name '*.py'`, `ls -R`
Read files with read_file before editing.
Edit only with edit_file (old_string must match exactly, no line numbers), create files with write_file.
Do not use sed/echo/cat to edit. Verify changes by running tests with bash. Be concise.
Tool calls may be denied by permission prompts; if a call is denied, do not retry it unchanged — say so and ask the user or propose an alternative."""

def tool(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}

S, I = {"type": "string"}, {"type": "integer"}
TOOLS = [
    tool("read_file", "Read file with line numbers", {"path": S, "offset": I, "limit": I}, ["path"]),
    tool("write_file", "Create/overwrite file", {"path": S, "content": S}, ["path", "content"]),
    tool("edit_file", "Replace old_string with new_string. old_string must occur exactly once.",
         {"path": S, "old_string": S, "new_string": S}, ["path", "old_string", "new_string"]),
    tool("bash", "Run shell command (find/grep/rg/ls/git/tests/builds)", {"command": S, "timeout": I}, ["command"]),
]

def trunc(s, n=30000):
    return s if len(s) <= n else s[:n] + f"\n... [truncated {len(s)-n} chars]"

def resolve(p):
    p = Path(p)
    p = p.resolve() if p.is_absolute() else (ROOT / p).resolve()
    if p != ROOT and ROOT not in p.parents:
        raise ValueError(f"outside workspace: {p}")
    return p

def read_file(a):
    p = resolve(a["path"])
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, int(a.get("offset", 1)))
    chunk = lines[start-1:start-1+int(a.get("limit", 2000))]
    return "\n".join(f"{i:5d} | {l}" for i, l in enumerate(chunk, start)) or "empty / out of range"

def write_file(a):
    p = resolve(a["path"]); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(a["content"], encoding="utf-8")
    return f"Wrote {len(a['content'])} chars to {p.relative_to(ROOT)}"

def edit_file(a):
    p = resolve(a["path"]); text = p.read_text(encoding="utf-8")
    if text.count(a["old_string"]) != 1:
        raise ValueError(f"old_string occurs {text.count(a['old_string'])} times, need exactly 1")
    p.write_text(text.replace(a["old_string"], a["new_string"], 1), encoding="utf-8")
    return f"Edited {p.relative_to(ROOT)}"

def bash(a):
    r = subprocess.run(a["command"], shell=True, executable=BASH, cwd=ROOT,
                       capture_output=True, text=True, timeout=int(a.get("timeout", 120)))
    return trunc(f"$ {a['command']}\n[exit {r.returncode}]\n{r.stdout}\n{r.stderr}")

HANDLERS = {"read_file": read_file, "write_file": write_file, "edit_file": edit_file, "bash": bash}

# --- Permissions -----------------------------------------------------------
# Modes, like Claude Code's permission modes (canonical names first):
#   default      - prompt for edits and non-read-only shell (legacy alias: ask)
#   accept-edits - file edits auto-approved, non-read-only shell still prompts
#                  (aliases: acceptEdits, approve-edits)
#   plan         - read-only only; edits and non-read-only shell are denied
#                  (legacy alias: deny; dont-ask)
#   auto         - Codex-style auto-review ("approve for me"): gated actions
#                  are sent to a separate reviewer model, which approves or
#                  denies them; reviewer failure/unparsable verdict denies.
#                  (aliases: approve-for-me, auto-review, guardian)
#   yolo         - allow everything, never prompt, never review
#                  (aliases: bypass-permissions, --yes)
# Non-interactive stdin in default mode fails closed (denies).
# Switch at runtime with /mode ..., or Ctrl+Tab / Shift+Tab at the REPL prompt.

PERMISSION_MODES = {"default", "accept-edits", "plan", "auto", "yolo"}
MODE_ALIASES = {
    "ask": "default", "normal": "default",
    "approve-edits": "accept-edits", "approve_edits": "accept-edits",
    "acceptedits": "accept-edits", "accept_edits": "accept-edits",
    "acceptEdits": "accept-edits",
    "deny": "plan", "dontask": "plan", "dont-ask": "plan", "dontAsk": "plan",
    "approve-for-me": "auto", "approve_for_me": "auto", "approveForMe": "auto",
    "auto-review": "auto", "auto_review": "auto", "autoReview": "auto",
    "guardian": "auto",
    "bypasspermissions": "yolo", "bypass-permissions": "yolo",
    "bypassPermissions": "yolo",
}
MODE_DESCRIPTIONS = {
    "default": "prompt for edits and non-read-only shell commands",
    "accept-edits": "file edits auto-approved; shell commands still prompt",
    "plan": "read-only: edits and non-read-only shell commands denied",
    "auto": "gated actions reviewed by a reviewer model (approve-for-me)",
    "yolo": "all tools auto-approved (bypass permissions)",
}
# Ctrl+Tab / Shift+Tab cycle order (Claude Code order, plus yolo last).
MODE_CYCLE = ["default", "accept-edits", "plan", "auto", "yolo"]

# Reviewer policy for `auto` mode, modelled on Codex's guardian/auto-review:
# block exfiltration, credential probing, persistent security weakening, and
# destructive/irreversible actions; allow ordinary in-workspace work.
AUTO_REVIEW_SYSTEM = """You are a permission reviewer (guardian) for a coding agent.
Decide whether the proposed tool action should be approved.
Deny: sending private data/secrets/credentials to untrusted destinations,
probing for credentials/tokens/cookies/session material, broad or persistent
weakening of security, and destructive actions with significant risk of
irreversible damage. Approve: ordinary, reversible, in-workspace coding work
that matches the user's evident task.
Reply with ONLY a JSON object: {"decision": "approve"|"deny", "rationale": "..."}.
If unsure, deny. Fail closed."""

def normalize_mode(mode):
    """Canonical mode name for `mode` (case-insensitive, aliases resolved)."""
    key = str(mode or "").strip()
    if key.lower() in PERMISSION_MODES:
        return key.lower()
    aliased = MODE_ALIASES.get(key) or MODE_ALIASES.get(key.lower())
    if aliased:
        return aliased
    raise ValueError(
        f"invalid permission mode: {mode!r} (want {'|'.join(MODE_CYCLE)}, "
        "e.g. --permission-mode accept-edits or AGENT_PERMISSION_MODE=auto)")

def next_mode(current, delta=1):
    """Mode `delta` steps from `current` along MODE_CYCLE (wraps around)."""
    idx = MODE_CYCLE.index(normalize_mode(current))
    return MODE_CYCLE[(idx + delta) % len(MODE_CYCLE)]

# Conservative read-only bash classifier. Anything it cannot prove read-only
# is prompted for. In particular, shell metacharacters (pipes, chaining,
# redirection, substitution) always force a prompt.
_READ_ONLY_COMMANDS = {"ls", "rg", "grep", "cat", "head", "tail", "wc", "pwd",
                       "diff", "stat", "file", "find", "git"}
_GIT_READ_ONLY = {"status", "log", "diff", "show", "blame", "describe",
                  "rev-parse", "ls-files"}
_FIND_DENIED = {"-exec", "-execdir", "-delete", "-ok", "-okdir"}

def is_read_only_bash(command):
    cmd = str(command or "").strip()
    if not cmd or any(c in cmd for c in ";\n\r|&<>$`"):
        return False
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return False
    if not tokens:
        return False
    head = tokens[0]
    if "/" in head:  # only absolute system paths, never ./local scripts
        if not head.startswith(("/bin/", "/usr/bin/")):
            return False
        head = Path(head).name
    if head not in _READ_ONLY_COMMANDS:
        return False
    if any(t.startswith("--output") or t == "--ext-diff" for t in tokens):
        return False
    if head == "git":
        if len(tokens) < 2 or tokens[1] not in _GIT_READ_ONLY:
            return False
    if head == "find" and any(
        t in _FIND_DENIED or t.startswith(("-fprint", "-fprintf", "-fls"))
        for t in tokens
    ):
        return False
    return True

def _pattern_for(name, args):
    """Session-allowlist key: exact bash command or exact file path."""
    if name == "bash":
        return str(args.get("command", "")).strip() or None
    if name in {"write_file", "edit_file", "read_file"}:
        return str(args.get("path", "")).strip() or None
    return None

def _preview(name, args):
    if name == "bash":
        return f"bash: {args.get('command', '')}"
    if name == "write_file":
        return f"write_file: {args.get('path', '')}\n{trunc(str(args.get('content', '')), 1000)}"
    if name == "edit_file":
        return (f"edit_file: {args.get('path', '')}\n"
                f"--- old_string ---\n{trunc(str(args.get('old_string', '')), 1000)}\n"
                f"--- new_string ---\n{trunc(str(args.get('new_string', '')), 1000)}")
    return f"{name}: {trunc(json.dumps(args, default=str), 1000)}"

def _is_tty():
    try:
        return sys.stdin.isatty()
    except Exception:
        return False

class Permissions:
    def __init__(self, mode=None, input_fn=None, isatty_fn=None, reviewer_fn=None):
        self.mode = normalize_mode(mode or os.environ.get("AGENT_PERMISSION_MODE") or "default")
        self.allowlist = set()  # {(tool_name, exact command/path)}
        self._input = input_fn or input
        self._isatty = isatty_fn or _is_tty
        # reviewer_fn(name, args, context) -> bool | (bool, rationale); used
        # only in `auto` mode. None means the default model-backed reviewer.
        self._reviewer = reviewer_fn

    def set_mode(self, mode):
        self.mode = normalize_mode(mode)
        return self.mode

    def cycle_mode(self, delta=1):
        """Advance (or, with delta=-1, rewind) through MODE_CYCLE."""
        self.mode = next_mode(self.mode, delta)
        return self.mode

    def describe(self):
        return MODE_DESCRIPTIONS[self.mode]

    def check(self, name, args, context=None):
        """Return True to allow the tool call, False to deny it."""
        if name == "read_file":
            return True
        if name == "bash" and is_read_only_bash(args.get("command", "")):
            return True
        if self.mode == "yolo":
            return True
        if self.mode == "plan":
            return False
        if self.mode == "accept-edits" and name in {"write_file", "edit_file"}:
            return True
        pattern = _pattern_for(name, args)
        if pattern is not None and (name, pattern) in self.allowlist:
            return True
        if self.mode == "auto":
            return self._auto_review(name, args, context)
        if not self._isatty():
            print(f"\n[permission] denied {name}: stdin is not a tty in {self.mode} mode. "
                  "Re-run with --yes (or AGENT_PERMISSION_MODE=yolo) to bypass permissions, "
                  "--permission-mode auto to have a reviewer model approve, "
                  "--permission-mode accept-edits to auto-approve edits, "
                  "or run interactively to be prompted.")
            return False
        return self._ask(name, args, pattern)

    def _auto_review(self, name, args, context=None):
        """Codex-style auto-review: a separate reviewer model decides.

        Fails closed: any reviewer error or unparsable verdict denies."""
        reviewer = self._reviewer or review_action  # default: model-backed
        try:
            verdict = reviewer(name, args, context)
        except Exception as e:
            print(f"\n[auto-review] denied {name}: reviewer unavailable ({e})")
            return False
        approved, rationale = verdict if isinstance(verdict, tuple) else (bool(verdict), "")
        print(f"\n[auto-review] {'approved' if approved else 'denied'} {name}"
              + (f": {rationale}" if rationale else ""))
        print(_preview(name, args))
        return approved

    def _ask(self, name, args, pattern):
        print(f"\n[permission] Allow this tool call?")
        print(_preview(name, args))
        print("[y]es once  [a]lways allow this exact command/path this session  [n]o")
        while True:
            try:
                ans = self._input("Allow? [y/a/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n[permission] denied (no input)")
                return False
            if ans in {"y", "yes"}:
                return True
            if ans in {"a", "always"}:
                if pattern is not None:
                    self.allowlist.add((name, pattern))
                return True
            if ans in {"n", "no"}:
                return False
            print("Please enter y, a, or n.")

try:
    PERMS = Permissions()
    _PERM_ERROR = None
except ValueError as e:  # e.g. bad AGENT_PERMISSION_MODE; reported in main(), not at import
    PERMS = Permissions(mode="default")
    _PERM_ERROR = str(e)

# --- REPL line input: Ctrl+Tab / Shift+Tab mode cycling --------------------
# Terminals have no single Ctrl+Tab encoding, so several are recognized:
#   Shift+Tab    ESC [ Z            (Claude Code's cycle key)
#   Ctrl+Tab     ESC [ 27;5;9~ / ESC [ 9;5u / ESC [ 1;5I / ESC [ 27;5;9u (kitty/xterm)
#   Ctrl+Tab when the terminal cannot distinguish it from plain Tab arrives
#   as Tab (0x09) -- the REPL has no tab-completion, so Tab cycles too.
#   Ctrl+Shift+Tab / Shift+Ctrl+Tab (modifier 6/8) cycles backwards.
# The prompt always shows the current mode, so a cycle is visible instantly.

def mode_switch_direction(seq):
    """+1/-1 if escape sequence `seq` (bytes after ESC, e.g. b'[Z') cycles
    permission modes, else 0."""
    if seq == b"[Z":
        return 1  # Shift+Tab
    if not seq.startswith(b"["):
        return 0
    body = seq[1:]
    final = body[-1:] if body else b""
    try:
        params = [int(p) for p in body[:-1].split(b";") if p] if len(body) > 1 else []
    except ValueError:
        return 0
    # Tab-with-modifiers: final I/u, or the kitty "9" key in a ~-sequence.
    is_tab_key = final in (b"I", b"u") or (final == b"~" and params[-1:] == [9])
    if not is_tab_key or len(params) < 2:
        return 0
    if final == b"~" or (final == b"u" and params[0] == 27):
        modifier = params[-2]  # [code; mod~ / [27; mod; code]u
    else:
        modifier = params[-1]  # [code; mod]I / [code; mod]u
    if modifier == 2:    # Shift+Tab (kitty encoding, no Ctrl)
        return 1
    if modifier in (6, 8):  # Shift+Ctrl
        return -1
    if modifier in (5, 7):  # Ctrl (+Alt)
        return 1
    return 0

def _next_byte(in_buf, timeout=0.05):
    """Next byte from `in_buf`, or None if none arrives within `timeout`."""
    try:
        fd = in_buf.fileno()
    except Exception:
        return in_buf.read(1) or None
    try:
        if not select.select([fd], [], [], timeout)[0]:
            return None
    except Exception:
        pass
    return in_buf.read(1) or None

def _repl_prompt(perms):
    return f"[{perms.mode}] > "

def _read_line_raw(perms, in_buf, out, history=None):
    """Read one REPL line in raw mode (caller sets terminal raw mode).

    Handles Ctrl+Tab/Shift+Tab/Tab mode cycling, history (up/down), and
    basic editing (backspace/delete, Ctrl-U/W/A/E). Raises EOFError /
    KeyboardInterrupt like input()."""
    buf, cursor = [], 0
    hist, hist_idx, saved = list(history or []), None, ""

    def redraw():
        prompt = _repl_prompt(perms)
        out.write("\r\x1b[K" + prompt + "".join(buf))
        if cursor < len(buf):
            out.write(f"\x1b[{len(buf) - cursor}D")
        out.flush()

    def load(text):
        nonlocal buf, cursor
        buf, cursor = list(text), len(text)
        redraw()

    out.write(_repl_prompt(perms)); out.flush()
    while True:
        b = in_buf.read(1)
        if not b:
            raise EOFError
        if b in (b"\r", b"\n"):
            out.write("\r\n"); out.flush()
            return "".join(buf)
        if b == b"\x03":  # Ctrl-C
            raise KeyboardInterrupt
        if b == b"\x04":  # Ctrl-D
            if not buf:
                raise EOFError
            if cursor < len(buf):
                del buf[cursor]; redraw()
            continue
        if b == b"\t":
            perms.cycle_mode(1); redraw(); continue
        if b in (b"\x7f", b"\x08"):  # Backspace
            if cursor:
                del buf[cursor - 1]; cursor -= 1; redraw()
            continue
        if b == b"\x15":  # Ctrl-U
            buf, cursor = [], 0; redraw(); continue
        if b == b"\x17":  # Ctrl-W: delete word before cursor
            while cursor and buf[cursor - 1] == " ": del buf[cursor - 1]; cursor -= 1
            while cursor and buf[cursor - 1] != " ": del buf[cursor - 1]; cursor -= 1
            redraw(); continue
        if b == b"\x01":  # Ctrl-A
            cursor = 0; redraw(); continue
        if b == b"\x05":  # Ctrl-E
            cursor = len(buf); redraw(); continue
        if b == b"\x1b":  # escape sequence
            second = _next_byte(in_buf)
            if second is None:
                continue
            if second == b"[":
                seq = b"["
                while True:
                    c = _next_byte(in_buf)
                    if c is None:
                        break
                    seq += c
                    if b"@" <= c <= b"~":  # final byte
                        break
                direction = mode_switch_direction(seq)
                if direction:
                    perms.cycle_mode(direction); redraw(); continue
                if seq == b"[A":  # up: history
                    if hist:
                        if hist_idx is None:
                            saved, hist_idx = "".join(buf), len(hist) - 1
                        elif hist_idx > 0:
                            hist_idx -= 1
                        load(hist[hist_idx])
                elif seq == b"[B":  # down: history
                    if hist_idx is not None:
                        if hist_idx < len(hist) - 1:
                            hist_idx += 1; load(hist[hist_idx])
                        else:
                            hist_idx = None; load(saved)
                elif seq == b"[C":  # right
                    cursor = min(len(buf), cursor + 1); redraw()
                elif seq == b"[D":  # left
                    cursor = max(0, cursor - 1); redraw()
                elif seq in (b"[H", b"[1~"):
                    cursor = 0; redraw()
                elif seq in (b"[F", b"[4~"):
                    cursor = len(buf); redraw()
                elif seq == b"[3~":  # delete
                    if cursor < len(buf):
                        del buf[cursor]; redraw()
                continue
            if second == b"O":
                _next_byte(in_buf)  # SS3: ignore
                continue
            continue  # Alt+key / lone ESC: ignore
        if b < b" ":  # other control chars: ignore
            continue
        # Printable (possibly multi-byte UTF-8).
        raw = b
        while b >= b"\xc0" or (raw != b and b >= b"\x80"):
            try:
                ch = raw.decode("utf-8"); break
            except UnicodeDecodeError:
                nxt = _next_byte(in_buf)
                if nxt is None:
                    ch = ""; break
                raw += nxt; b = nxt
        else:
            ch = raw.decode("utf-8", errors="replace")
        if ch:
            buf.insert(cursor, ch); cursor += 1; redraw()

class _FdReader:
    """Unbuffered byte reader over a raw fd (os.read per byte).

    sys.stdin.buffer must not be used here: it prefetches bytes into a
    Python buffer, so select() on the fd never sees them and escape
    sequences would leak into the line as literal text."""

    def __init__(self, fd):
        self.fd = fd

    def read(self, n=-1):
        return os.read(self.fd, 1 if n == 1 else 4096)

    def fileno(self):
        return self.fd

def read_repl_line(perms, history=None):
    """Read a task line from the REPL, cycling permission modes on Ctrl+Tab.

    Falls back to plain input() without a tty / termios (no cycling)."""
    if termios is None or not _is_tty():
        return input(_repl_prompt(perms))
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return _read_line_raw(perms, _FdReader(fd), sys.stdout, history)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="nada.py", description="Minimal coding agent with permission prompts.")
    p.add_argument("--yes", "-y", "--auto-approve", action="store_true",
                   help="allow all tool calls without prompting or review (same as --permission-mode yolo)")
    p.add_argument("--permission-mode", "--mode", dest="permission_mode",
                   choices=sorted(PERMISSION_MODES | set(MODE_ALIASES)), default=None,
                   metavar="MODE",
                   help="permission mode: default|accept-edits|plan|auto|yolo "
                        "(auto = reviewer-model approve-for-me; "
                        "aliases: ask, approve-edits, acceptEdits, deny, approve-for-me, "
                        "auto-review, bypass-permissions; "
                        "default: $AGENT_PERMISSION_MODE or default)")
    p.add_argument("--effort", "--reasoning-effort", dest="effort", default=None,
                   metavar="LEVEL",
                   help="model reasoning effort level, discovered per-model from OpenRouter "
                        "(e.g. max|xhigh|high|medium|low|minimal for meta/muse-ember; "
                        "which levels exist depends on MODEL; 'default' = model default; "
                        "default: $AGENT_EFFORT)")
    p.add_argument("--accept-edits", action="store_true",
                   help="auto-approve file edits; shell commands still prompt")
    p.add_argument("--plan", action="store_true",
                   help="read-only mode: deny edits and non-read-only shell commands")
    session_group = p.add_mutually_exclusive_group()
    session_group.add_argument("--session", dest="session", metavar="ID", default=None,
                               help="resume the saved session ID (or 'latest') instead of "
                                    "starting a new one; unknown IDs are an error")
    session_group.add_argument("--continue", dest="continue_session", action="store_true",
                               help="resume the most recently saved session")
    p.add_argument("--session-dir", dest="session_dir", metavar="DIR", default=None,
                   help="directory for saved sessions (default: $AGENT_SESSION_DIR or ~/.nada/sessions)")
    p.add_argument("--no-autosave", dest="no_autosave", action="store_true",
                   help="do not auto-save the session after each turn (REPL /session save still works)")
    p.add_argument("task", nargs="*", help="task to run non-interactively; omit for the REPL")
    return p.parse_args(argv)

def _parse_retry_after(value):
    """`Retry-After` header -> seconds to wait, or None if absent/unparsable.

    Handles both delta-seconds ("5") and HTTP-date forms."""
    if not value:
        return None
    value = str(value).strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())

def _backoff_delay(attempt, retry_after=None):
    """Wait before retry `attempt` (0-based): RETRY_BASE_DELAY *
    RETRY_BACKOFF ** attempt (1s, 2s, 4s, ... by default), capped at
    RETRY_MAX_DELAY, with equal jitter (50-100% of the nominal wait, so
    concurrent clients don't retry in lockstep). A server Retry-After is
    a floor: never retry sooner than it asks (it may exceed the cap)."""
    nominal = min(RETRY_MAX_DELAY, RETRY_BASE_DELAY * RETRY_BACKOFF ** attempt)
    delay = nominal / 2 + random.uniform(0, nominal / 2)
    return max(delay, retry_after) if retry_after is not None else delay

def call_model(messages, model=None, use_tools=True, *, max_retries=None, sleep_fn=None):
    """POST one chat completion. HTTP 429 (rate limit) is retried up to
    `max_retries` times with exponential backoff (see _backoff_delay);
    any other HTTP error raises immediately. Both the main agent and
    the `auto`-mode reviewer go through here, so both get the retries.
    The session effort, if set, is sent as `reasoning: {"effort": ...}`
    for the main model."""
    retries = max(0, MAX_RETRIES if max_retries is None else max_retries)
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn
    for attempt in range(retries + 1):
        body = {"model": model or MODEL, "messages": messages}
        # Effort applies to the main model only: a different reviewer model
        # uses its own default rather than risking an unsupported level.
        if EFFORT and body["model"] == MODEL:
            body["reasoning"] = {"effort": EFFORT}
        if use_tools:
            body.update({"tools": TOOLS, "tool_choice": "auto"})
        payload = json.dumps(body).encode()
        req = urllib.request.Request(API_URL, data=payload, method="POST", headers={
            "Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json",
            "X-Title": "stdlib-agent"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            e.close()  # release the connection before sleeping / retrying
            if e.code != 429 or attempt == retries:
                raise RuntimeError(f"OpenRouter {e.code}: {detail}") from e
            headers = getattr(e, "headers", None)
            delay = _backoff_delay(attempt, _parse_retry_after(
                headers.get("Retry-After") if headers else None))
            print(f"[rate-limit] OpenRouter 429; retrying in {delay:.1f}s "
                  f"({attempt + 1}/{retries})", file=sys.stderr)
            sleep_fn(delay)

def _parse_review_verdict(text):
    """(approved, rationale) from a reviewer reply; unparsable denies."""
    text = (text or "").strip()
    try:
        obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
        decision, rationale = str(obj.get("decision", "")).lower(), str(obj.get("rationale", ""))
    except Exception:
        decision, rationale = text.lower(), text
    if decision in {"approve", "approved", "allow", "allowed", "yes"}:
        return True, rationale
    return False, rationale or "reviewer did not approve"

def review_action(name, args, context=None):
    """Default `auto`-mode reviewer: ask REVIEW_MODEL to approve/deny.

    Returns (approved, rationale). Raises on transport/HTTP errors so
    Permissions._auto_review can fail closed."""
    messages = [
        {"role": "system", "content": AUTO_REVIEW_SYSTEM},
        {"role": "user", "content": (
            f"Proposed action:\n{_preview(name, args)}"
            + (f"\n\nContext (recent conversation):\n{trunc(str(context), 4000)}" if context else ""))},
    ]
    reply = call_model(messages, model=REVIEW_MODEL, use_tools=False)
    content = reply["choices"][0]["message"].get("content") or ""
    return _parse_review_verdict(content)

def run_turns(messages):
    for _ in range(MAX_TURNS):
        msg = call_model(messages)["choices"][0]["message"]
        messages.append(msg)  # verbatim — required for tool_calls validation
        if msg.get("content"): print(msg["content"])
        calls = msg.get("tool_calls") or []
        if not calls: return
        for call in calls:
            fn = call["function"]; name = fn["name"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception as e:
                result = f"Error: {type(e).__name__}: {e}"
                print(f"\n[{name}] invalid arguments\n{result}")
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": trunc(result)})
                continue
            if name not in HANDLERS:
                result = f"Error: unknown tool: {name}"
            elif not PERMS.check(name, args, context=trunc(json.dumps(messages[-6:], default=str), 4000)):
                result = f"Error: PermissionDenied: denied {name} (permission mode: {PERMS.mode})"
            else:
                try:
                    result = HANDLERS[name](args)
                except Exception as e:
                    result = f"Error: {type(e).__name__}: {e}"
            print(f"\n[{name}] {fn.get('arguments')}\n{trunc(result, 2000)}")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": trunc(result)})
    print(f"Stopped after {MAX_TURNS} turns")

def handle_permissions_command(parts, perms=None):
    """Handle `/mode ...` and `/permissions ...` (same options; `/mode clear`
    is not allowed -- clearing is a /permissions thing)."""
    perms = perms if perms is not None else PERMS
    if len(parts) == 1:
        print(f"Permissions: mode={perms.mode} ({perms.describe()})")
        print("Modes: " + ", ".join(MODE_CYCLE) + "  (Ctrl+Tab / Shift+Tab cycles)")
        print("Session allowlist:")
        for name, pattern in sorted(perms.allowlist):
            print(f"  {name}: {pattern}")
        if not perms.allowlist:
            print("  (empty)")
    elif parts[1] == "clear" and parts[0] == "/permissions":
        perms.allowlist.clear()
        print("Session allowlist cleared")
    else:
        try:
            mode = perms.set_mode(parts[1])
        except ValueError:
            print("Usage: /mode [default|accept-edits|plan|auto|yolo]   "
                  "(aliases: ask, approve-edits, acceptEdits, deny, approve-for-me, bypass-permissions)")
            print("       /permissions [default|accept-edits|plan|auto|yolo|clear]")
        else:
            print(f"Permissions mode: {mode} ({perms.describe()})")

# /mode is the Claude-style spelling; /permissions keeps its allowlist/clear.
handle_mode_command = handle_permissions_command

def handle_effort_command(parts):
    """Handle `/effort`: show the current effort and the levels discovered
    for MODEL; `/effort LEVEL` sets it; `/effort clear` (or default/reset)
    unsets it back to the model default."""
    if len(parts) == 1:
        try:
            info = fetch_model_reasoning()
        except Exception as e:
            print(_effort_summary(None))
            print(f"  (discovery failed: {e})")
        else:
            print(_effort_summary(info))
            if info["supported_efforts"]:
                print(f"Usage: /effort [{'|'.join(info['supported_efforts'])}|clear]")
        return
    if len(parts) != 2:
        print("Usage: /effort [LEVEL|clear]   (no argument shows available levels)")
        return
    if normalize_effort(parts[1]) is None:
        set_effort(None)
        print(f"Effort cleared: model default for {MODEL}")
        return
    try:
        supported = discover_efforts()
    except Exception as e:  # offline etc.: validate against gateway levels only
        print(f"Warning: could not discover effort levels for {MODEL} ({e}); "
              "validating against all gateway levels only")
        supported = None
    try:
        level = set_effort(parts[1], supported)
    except ValueError as e:
        print(f"Invalid effort: {e}  (usage: /effort [LEVEL|clear])")
        return
    print(f"Effort set: {level} for {MODEL}")

# --- Sessions --------------------------------------------------------------
# A session is the persisted conversation state: messages (stored verbatim,
# so tool_calls survive a restart), REPL history, and the permission
# mode/allowlist and effort in force. One JSON file per session under
# ~/.nada/sessions (override with --session-dir / AGENT_SESSION_DIR).
# Sessions are autosaved after every REPL turn and after one-shot runs,
# so quitting and re-running with --session ID / --continue resumes where
# you left off; `/session save` forces a save (e.g. with --no-autosave).

SESSION_VERSION = 1
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_SESSION_USAGE = ("Usage: /session [save [TITLE]|list|resume ID|new|delete ID|path]  "
                  "(aliases: /sessions, /resume ID)")

def default_session_dir():
    """Directory saved sessions live in: $AGENT_SESSION_DIR (alias
    $NADA_SESSION_DIR) or ~/.nada/sessions."""
    env = os.environ.get("AGENT_SESSION_DIR") or os.environ.get("NADA_SESSION_DIR")
    if env:
        return Path(env).expanduser()
    try:
        return Path.home() / ".nada" / "sessions"
    except Exception:
        return Path.cwd() / ".nada" / "sessions"

def _new_session_id():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]

def sanitize_session_id(session_id):
    """A session id is a plain file stem; anything else (paths, `..`,
    separators) is rejected so ids can never escape the session dir."""
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(f"invalid session id: {session_id!r}")
    return session_id

def _utcnow():
    return datetime.now(timezone.utc).isoformat()

def _derive_title(messages):
    """Title from the first user message, or 'untitled session'."""
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
            return " ".join(m["content"].strip().splitlines()[0].split())[:60]
    return "untitled session"

def _count_turns(messages):
    return sum(1 for m in messages if m.get("role") == "user")

def _trim_incomplete_tool_calls(messages):
    """Longest prefix with no dangling tool_calls: an assistant message
    whose tool results never arrived (interrupted mid-turn) would make the
    next API call fail, so it is dropped along with anything after it."""
    pending, last_clean = set(), 0
    for idx, m in enumerate(messages):
        if m.get("role") == "assistant":
            for call in m.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id"):
                    pending.add(call["id"])
        elif m.get("role") == "tool":
            pending.discard(m.get("tool_call_id"))
        if not pending:
            last_clean = idx + 1
    return [dict(m) for m in messages[:last_clean]]

def _validate_session_data(data):
    """Normalized session dict, or ValueError if the data is not a
    well-formed session of the current version."""
    if not isinstance(data, dict):
        raise ValueError("session file must contain a JSON object")
    if data.get("version") != SESSION_VERSION:
        raise ValueError(f"unsupported session version: {data.get('version')!r}")
    sid = sanitize_session_id(data.get("id"))
    messages = data.get("messages")
    if (not isinstance(messages, list) or not messages or
            any(not isinstance(m, dict) or not m.get("role") for m in messages)):
        raise ValueError("session has no valid messages")
    history = data.get("history") or []
    if not isinstance(history, list) or any(not isinstance(h, str) for h in history):
        raise ValueError("session history must be a list of strings")
    allowlist = data.get("allowlist") or []
    if (not isinstance(allowlist, list) or
            any(not (isinstance(e, (list, tuple)) and len(e) == 2 and
                     all(isinstance(x, str) for x in e)) for e in allowlist)):
        raise ValueError("session allowlist must be [tool, pattern] pairs")
    out = dict(data)
    out.update({"id": sid, "messages": [dict(m) for m in messages],
                "history": list(history),
                "allowlist": [[name, pattern] for name, pattern in allowlist]})
    return out

def session_summary(data):
    return {"id": data["id"], "title": data.get("title") or "untitled session",
            "updated_at": data.get("updated_at") or "",
            "turns": _count_turns(data.get("messages") or []),
            "messages": len(data.get("messages") or [])}

class SessionStore:
    """JSON-file storage for sessions: one `<id>.json` per session."""

    def __init__(self, directory=None):
        self.dir = Path(directory).expanduser() if directory else default_session_dir()

    def path_for(self, session_id):
        return self.dir / f"{sanitize_session_id(session_id)}.json"

    def save(self, data):
        """Validate and atomically write a session; returns (path, saved data)."""
        data = _validate_session_data(data)
        data["updated_at"] = _utcnow()
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(data["id"])
        tmp = self.dir / f".{data['id']}.json.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        try:
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(tmp, path)  # atomic: a crash never leaves a half-written session
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except TypeError:  # older Pythons: unlink() has no missing_ok
                if tmp.exists():
                    tmp.unlink()
        return path, data

    def _load_file(self, path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"corrupt session {path.name}: {e}") from e
        return _validate_session_data(data)

    def load(self, session_id):
        """Load a session by id ('latest' = most recently updated)."""
        if session_id == "latest":
            summaries = self.list()
            if not summaries:
                raise FileNotFoundError("no saved sessions")
            session_id = summaries[0]["id"]
        path = self.path_for(session_id)
        if not path.is_file():
            raise FileNotFoundError(f"session not found: {session_id}")
        return self._load_file(path)

    def list(self):
        """Summaries of all saved sessions, most recently updated first.
        Corrupt files are skipped with a warning, not fatal."""
        if not self.dir.is_dir():
            return []
        summaries = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                summaries.append(session_summary(self._load_file(path)))
            except Exception as e:
                print(f"Warning: skipping {path.name}: {e}", file=sys.stderr)
        summaries.sort(key=lambda s: s["updated_at"], reverse=True)
        return summaries

    def delete(self, session_id):
        path = self.path_for(session_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

class Session:
    """Live session: messages/history for run_turns() plus metadata to
    persist/restore them. PERMS/EFFORT go through the module globals, so a
    restored session brings back its mode, allowlist, and effort too."""

    def __init__(self, store, *, session_id=None, title=None, messages=None, history=None):
        self.store = store
        self.id = sanitize_session_id(session_id) if session_id else _new_session_id()
        self.title = title
        self.created_at = self.updated_at = _utcnow()
        self.root, self.model = str(ROOT), MODEL
        self.messages = messages if messages is not None else [{"role": "system", "content": SYSTEM}]
        self.history = history if history is not None else []
        self.autosave = True

    @property
    def turns(self):
        return _count_turns(self.messages)

    def path(self):
        return self.store.path_for(self.id)

    def snapshot(self):
        """Persistable form of the current state (dangling tool_calls trimmed)."""
        return {"version": SESSION_VERSION, "id": self.id,
                "title": self.title or _derive_title(self.messages),
                "created_at": self.created_at, "updated_at": self.updated_at,
                "root": self.root, "model": self.model, "effort": EFFORT,
                "permission_mode": PERMS.mode,
                "allowlist": sorted([list(item) for item in PERMS.allowlist]),
                "history": list(self.history),
                "messages": _trim_incomplete_tool_calls(self.messages)}

    def save(self):
        """Save now (even with autosave off); returns the file path."""
        path, data = self.store.save(self.snapshot())
        self.title, self.updated_at = data["title"], data["updated_at"]
        return path

    def try_save(self):
        """Best-effort autosave: only once there is a turn to save, and a
        failure warns instead of crashing the REPL."""
        if not self.autosave or not self.turns:
            return False
        try:
            self.save()
        except Exception as e:
            print(f"Warning: could not save session {self.id} ({e})", file=sys.stderr)
            return False
        return True

    def restore(self, data, *, apply_effort=True, supported_efforts=None):
        """Replace messages/history/effort with a saved session's. The
        system prompt is regenerated for the current ROOT; a different
        originating root is reported, not fatal. Permission mode and
        allowlist are not restored (see below)."""
        data = _validate_session_data(data)
        self.id, self.title = data["id"], data.get("title")
        self.created_at = data.get("created_at") or _utcnow()
        self.updated_at = data.get("updated_at") or self.created_at
        self.root, self.model = data.get("root") or str(ROOT), data.get("model") or MODEL
        messages = [dict(m) for m in data["messages"]]
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": SYSTEM}
        else:
            messages.insert(0, {"role": "system", "content": SYSTEM})
        self.messages[:] = _trim_incomplete_tool_calls(messages)
        self.history[:] = list(data.get("history") or [])
        # Permission mode/allowlist are recorded in the file but deliberately
        # not restored: loading a file must not silently escalate permissions.
        if data.get("permission_mode") and data["permission_mode"] != PERMS.mode:
            print(f"Note: session was saved with permission mode {data['permission_mode']} "
                  f"(current: {PERMS.mode}; not restored -- use /mode to change)")
        if apply_effort and "effort" in data:
            try:
                set_effort(data.get("effort"), supported_efforts)
            except ValueError as e:
                print(f"Warning: saved effort {data.get('effort')!r} not applied ({e})")
        if data.get("root") and data["root"] != str(ROOT):
            print(f"Note: session was created in {data['root']}; current root is {ROOT}")
        return self

    def reset(self):
        """Start a new, empty session (new id, fresh history)."""
        self.id, self.title = _new_session_id(), None
        self.created_at = self.updated_at = _utcnow()
        self.messages[:] = [{"role": "system", "content": SYSTEM}]
        self.history[:] = []

def handle_session_command(parts, session):
    """Handle `/session ...` (and the `/sessions` / `/resume ID` aliases)."""
    cmd, rest = parts[0], parts[1:]
    if cmd == "/sessions":
        rest = ["list"]
    elif cmd == "/resume":
        rest = ["resume"] + rest
    if not rest:
        print(f"Session: {session.id}")
        print(f"Title: {session.title or _derive_title(session.messages)}")
        print(f"Turns: {session.turns}  Messages: {len(session.messages)}")
        print(f"Saved to: {session.path()}")
        print(_SESSION_USAGE)
        return
    sub, args = rest[0], rest[1:]
    if sub in {"list", "ls"}:
        summaries = session.store.list()
        if not summaries:
            print("No saved sessions")
        for s in summaries:
            print(f"{s['id']}  {s['updated_at']}  turns={s['turns']}  {s['title']}")
    elif sub == "save":
        if args:
            session.title = " ".join(args)
        try:
            path = session.save()
        except Exception as e:
            print(f"Could not save session: {e}")
        else:
            print(f"Session saved: {session.id} ({path})")
    elif sub == "new":
        session.reset()
        print(f"New session: {session.id}")
    elif sub == "path":
        print(session.path())
    elif sub in {"resume", "load", "continue"}:
        target = args[0] if args else ("latest" if sub == "continue" else None)
        if target is None:
            print(_SESSION_USAGE)
            return
        try:
            data = session.store.load(target)
        except FileNotFoundError:
            print(f"Session not found: {target}")
        except Exception as e:
            print(f"Could not resume session {target}: {e}")
        else:
            session.restore(data)
            print(f"Resumed session: {session.id} ({session.turns} turns)")
    elif sub == "delete":
        if not args:
            print(_SESSION_USAGE)
            return
        try:
            deleted = session.store.delete(args[0])
        except ValueError as e:
            print(f"Could not delete session: {e}")
            return
        if not deleted:
            print(f"Session not found: {args[0]}")
            return
        print(f"Deleted session: {args[0]}")
        if args[0] == session.id:  # don't let autosave resurrect the live session
            session.reset()
            print(f"New session: {session.id}")
    else:
        print(_SESSION_USAGE)

def main():
    args = parse_args()
    if _PERM_ERROR and not (args.yes or args.permission_mode or args.accept_edits or args.plan):
        sys.exit(f"error: {_PERM_ERROR}")
    if args.accept_edits:
        PERMS.set_mode("accept-edits")
    elif args.plan:
        PERMS.set_mode("plan")
    if args.yes:
        PERMS.set_mode("yolo")
    elif args.permission_mode:
        PERMS.set_mode(args.permission_mode)
    if args.effort is not None:
        try:
            set_effort(args.effort)  # validated against the discovered levels below
        except ValueError as e:
            sys.exit(f"error: {e}")
    if not API_KEY: sys.exit("set OPENROUTER_API_KEY (and optionally MODEL)")
    # Discover this model's effort levels from OpenRouter (best-effort: a
    # network failure only loses local validation, not the session).
    try:
        _reasoning = fetch_model_reasoning()
    except Exception as e:
        print(f"Warning: could not discover effort levels for {MODEL} ({e}); "
              "continuing without per-model validation", file=sys.stderr)
        _reasoning = None
    if EFFORT is not None:
        try:
            validate_effort(EFFORT, _reasoning["supported_efforts"] if _reasoning else None)
        except ValueError as e:
            sys.exit(f"error: {e}")
    store = SessionStore(args.session_dir)
    session = Session(store)
    session.autosave = not args.no_autosave
    if args.session or args.continue_session:
        target = args.session or "latest"
        try:
            data = store.load(target)
        except FileNotFoundError:
            sys.exit(f"error: session not found: {target} (dir: {store.dir})")
        except Exception as e:
            sys.exit(f"error: could not load session {target}: {e}")
        session.restore(data, apply_effort=args.effort is None,
                        supported_efforts=_reasoning["supported_efforts"] if _reasoning else None)
        print(f"Resumed session: {session.id} ({session.turns} turns)")
    if args.task:
        session.messages.append({"role": "user", "content": " ".join(args.task)})
        try:
            run_turns(session.messages)
        finally:
            session.try_save()
        return
    print(f"Model: {MODEL}\n{_effort_summary(_reasoning)}\nRoot: {ROOT}\nPermissions: {PERMS.mode} ({PERMS.describe()})")
    print(f"Session: {session.id} ({session.path()})")
    print("/quit to exit, /clear to reset, /mode [default|accept-edits|plan|auto|yolo] to switch modes, "
          "/effort [LEVEL|clear] to set model effort, "
          "/session [save|list|resume ID|new] to manage sessions")
    print("Ctrl+Tab / Shift+Tab at the prompt cycles modes; /permissions shows the session allowlist")
    while True:
        print()
        try: task = read_repl_line(PERMS, session.history).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if task in {"/quit", "/exit"}: break
        if task == "/clear": session.messages[:] = [{"role": "system", "content": SYSTEM}]; continue
        if task.split() and task.split()[0] in {"/permissions", "/mode"}:
            handle_permissions_command(task.split()); continue
        if task.split() and task.split()[0] == "/effort":
            handle_effort_command(task.split()); continue
        if task.split() and task.split()[0] in {"/session", "/sessions", "/resume"}:
            handle_session_command(task.split(), session); continue
        if task:
            session.history.append(task)
            session.messages.append({"role": "user", "content": task})
            try: run_turns(session.messages)
            except Exception as e: print(f"Error: {e}")
            finally: session.try_save()
    session.try_save()

if __name__ == "__main__":
    main()
