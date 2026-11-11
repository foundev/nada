#!/usr/bin/env python3
import argparse, json, os, shlex, subprocess, sys, urllib.error, urllib.request
from pathlib import Path

ROOT = Path.cwd().resolve()
MODEL = os.environ.get("MODEL", "meta/muse-ember")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "2000000"))
API_URL = "https://openrouter.ai/api/v1/chat/completions"
BASH = "/bin/bash" if Path("/bin/bash").exists() else None

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
# Modes: ask (default), auto, deny.
#   ask  - prompt for mutating / shell tools; read-only tools run without prompting
#   auto - allow everything (same as --yes)
#   deny - allow only read-only tools, deny everything else
# Non-interactive stdin in ask mode fails closed (denies).

PERMISSION_MODES = {"ask", "auto", "deny"}

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
    def __init__(self, mode=None, input_fn=None, isatty_fn=None):
        mode = (mode or os.environ.get("AGENT_PERMISSION_MODE") or "ask").lower()
        if mode not in PERMISSION_MODES:
            raise ValueError(
                f"invalid permission mode: {mode!r} (want ask|auto|deny, "
                "e.g. --permission-mode auto or AGENT_PERMISSION_MODE=auto)")
        self.mode = mode
        self.allowlist = set()  # {(tool_name, exact command/path)}
        self._input = input_fn or input
        self._isatty = isatty_fn or _is_tty

    def check(self, name, args):
        """Return True to allow the tool call, False to deny it."""
        if name == "read_file":
            return True
        if name == "bash" and is_read_only_bash(args.get("command", "")):
            return True
        if self.mode == "auto":
            return True
        if self.mode == "deny":
            return False
        pattern = _pattern_for(name, args)
        if pattern is not None and (name, pattern) in self.allowlist:
            return True
        if not self._isatty():
            print(f"\n[permission] denied {name}: stdin is not a tty in ask mode. "
                  "Re-run with --yes (or AGENT_PERMISSION_MODE=auto) to auto-approve, "
                  "or run interactively to be prompted.")
            return False
        return self._ask(name, args, pattern)

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
    PERMS = Permissions(mode="ask")
    _PERM_ERROR = str(e)

def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="nada.py", description="Minimal coding agent with permission prompts.")
    p.add_argument("--yes", "-y", "--auto-approve", action="store_true",
                   help="auto-approve all tool calls (same as --permission-mode auto)")
    p.add_argument("--permission-mode", choices=sorted(PERMISSION_MODES), default=None,
                   metavar="ask|auto|deny",
                   help="permission mode (default: $AGENT_PERMISSION_MODE or ask)")
    p.add_argument("task", nargs="*", help="task to run non-interactively; omit for the REPL")
    return p.parse_args(argv)

def call_model(messages):
    payload = json.dumps({"model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto"}).encode()
    req = urllib.request.Request(API_URL, data=payload, method="POST", headers={
        "Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json",
        "X-Title": "stdlib-agent"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenRouter {e.code}: {e.read().decode(errors='replace')}") from e

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
            elif not PERMS.check(name, args):
                result = f"Error: PermissionDenied: user denied {name} (permission mode: {PERMS.mode})"
            else:
                try:
                    result = HANDLERS[name](args)
                except Exception as e:
                    result = f"Error: {type(e).__name__}: {e}"
            print(f"\n[{name}] {fn.get('arguments')}\n{trunc(result, 2000)}")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": trunc(result)})
    print(f"Stopped after {MAX_TURNS} turns")

def handle_permissions_command(parts):
    if len(parts) == 1:
        print(f"Permissions: mode={PERMS.mode}")
        print("Session allowlist:")
        for name, pattern in sorted(PERMS.allowlist):
            print(f"  {name}: {pattern}")
        if not PERMS.allowlist:
            print("  (empty)")
    elif parts[1] in PERMISSION_MODES:
        PERMS.mode = parts[1]
        print(f"Permissions mode: {PERMS.mode}")
    elif parts[1] == "clear":
        PERMS.allowlist.clear()
        print("Session allowlist cleared")
    else:
        print("Usage: /permissions [ask|auto|deny|clear]")

def main():
    args = parse_args()
    if _PERM_ERROR and not (args.yes or args.permission_mode):
        sys.exit(f"error: {_PERM_ERROR}")
    if args.yes:
        PERMS.mode = "auto"
    elif args.permission_mode:
        PERMS.mode = args.permission_mode
    if not API_KEY: sys.exit("set OPENROUTER_API_KEY (and optionally MODEL)")
    messages = [{"role": "system", "content": SYSTEM}]
    if args.task:
        messages.append({"role": "user", "content": " ".join(args.task)}); run_turns(messages); return
    print(f"Model: {MODEL}\nRoot: {ROOT}\nPermissions: {PERMS.mode} (read-only tools auto-allowed)")
    print("/quit to exit, /clear to reset, /permissions [ask|auto|deny|clear] to manage permissions")
    while True:
        try: task = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if task in {"/quit", "/exit"}: break
        if task == "/clear": messages = [{"role": "system", "content": SYSTEM}]; continue
        if task.startswith("/permissions"):
            handle_permissions_command(task.split()); continue
        if task:
            messages.append({"role": "user", "content": task})
            try: run_turns(messages)
            except Exception as e: print(f"Error: {e}")

if __name__ == "__main__":
    main()
