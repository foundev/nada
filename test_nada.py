#!/usr/bin/env python3
import io
import json
import time
import unittest
import urllib.error
from email.utils import formatdate
from unittest.mock import patch

import nada


def perms(mode="ask", answers=None, tty=True, reviewer=None):
    answers = iter(answers or [])
    def fake_input(prompt=""):
        try:
            return next(answers)
        except StopIteration:
            raise AssertionError("input() should not have been called")
    return nada.Permissions(mode=mode, input_fn=fake_input, isatty_fn=lambda: tty,
                            reviewer_fn=reviewer)


class TestReadOnlyBash(unittest.TestCase):
    def test_allowed(self):
        for cmd in [
            "ls -R", "rg -n PATTERN .", "grep -rn foo .", "cat nada.py",
            "head -20 nada.py", "tail -5 nada.py", "wc -l nada.py", "pwd",
            "diff a b", "stat nada.py", "find . -name '*.py'",
            "git status", "git log --oneline -3", "git diff", "git show HEAD",
            "/bin/ls -la", "/usr/bin/rg foo .",
        ]:
            self.assertTrue(nada.is_read_only_bash(cmd), cmd)

    def test_denied(self):
        for cmd in [
            "rm -rf build", "ls; rm -rf /", "ls && rm x", "ls | grep x",
            "cat x > y", "cat x >> y", "ls $(rm x)", "ls `rm x`", "ls $HOME",
            "echo hi > x", "python3 test.py", "pytest -q", "npm test",
            "git add .", "git commit -m x", "git push", "git branch -D x",
            "git diff --output=/tmp/x", "git log --ext-diff",
            "find . -exec rm {} \\;", "find . -delete", "find . -fprint /tmp/x",
            "sudo ls", "FOO=bar ls", "./script.sh", "/tmp/ls", "make build",
            "", "   ", "ls\nrm x",
        ]:
            self.assertFalse(nada.is_read_only_bash(cmd), repr(cmd))


class TestPermissions(unittest.TestCase):
    def test_read_file_always_allowed(self):
        for mode in ["ask", "auto", "yolo", "deny"]:
            self.assertTrue(perms(mode).check("read_file", {"path": "nada.py"}))

    def test_read_only_bash_always_allowed(self):
        for mode in ["ask", "auto", "yolo", "deny"]:
            self.assertTrue(perms(mode).check("bash", {"command": "ls -R"}))

    def test_yolo_allows_everything(self):
        p = perms("yolo")  # no reviewer/answers: any review/prompt would raise
        self.assertTrue(p.check("write_file", {"path": "x", "content": "y"}))
        self.assertTrue(p.check("bash", {"command": "rm -rf build"}))

    def test_auto_reviews_gated_actions(self):
        seen = []
        def reviewer(name, args, context=None):
            seen.append((name, args))
            return (name != "bash", "ok" if name != "bash" else "too destructive")
        p = perms("auto", reviewer=reviewer)
        self.assertTrue(p.check("write_file", {"path": "x", "content": "y"}))
        self.assertFalse(p.check("bash", {"command": "rm -rf build"}))
        self.assertEqual(seen, [("write_file", {"path": "x", "content": "y"}),
                                ("bash", {"command": "rm -rf build"})])

    def test_auto_skips_review_for_readonly_and_allowlist(self):
        def reviewer(name, args, context=None):
            raise AssertionError("reviewer should not be called")
        p = perms("auto", reviewer=reviewer)
        self.assertTrue(p.check("read_file", {"path": "nada.py"}))
        self.assertTrue(p.check("bash", {"command": "ls -R"}))
        p.allowlist.add(("bash", "make build"))
        self.assertTrue(p.check("bash", {"command": "make build"}))

    def test_auto_reviewer_failure_denies(self):
        def reviewer(name, args, context=None):
            raise RuntimeError("boom")
        p = perms("auto", reviewer=reviewer)
        self.assertFalse(p.check("bash", {"command": "make build"}))

    def test_parse_review_verdict(self):
        self.assertEqual(nada._parse_review_verdict('{"decision": "approve", "rationale": "fine"}'),
                         (True, "fine"))
        self.assertEqual(nada._parse_review_verdict('{"decision": "deny", "rationale": "no"}'),
                         (False, "no"))
        self.assertEqual(nada._parse_review_verdict("approve")[0], True)
        self.assertFalse(nada._parse_review_verdict("maybe")[0])  # unparsable -> deny
        self.assertFalse(nada._parse_review_verdict("")[0])

    def test_deny_blocks_mutating_and_shell(self):
        p = perms("deny")
        self.assertFalse(p.check("write_file", {"path": "x", "content": "y"}))
        self.assertFalse(p.check("edit_file", {"path": "x", "old_string": "a", "new_string": "b"}))
        self.assertFalse(p.check("bash", {"command": "rm -rf build"}))

    def test_ask_yes_once(self):
        p = perms("ask", ["y"])
        self.assertTrue(p.check("bash", {"command": "make build"}))
        self.assertEqual(p.allowlist, set())  # not remembered

    def test_ask_no(self):
        p = perms("ask", ["n"])
        self.assertFalse(p.check("write_file", {"path": "x", "content": "y"}))

    def test_ask_always_remembers_exact_command_only(self):
        p = perms("ask", ["a"])
        self.assertTrue(p.check("bash", {"command": "make build"}))
        self.assertIn(("bash", "make build"), p.allowlist)
        # Same exact command: no further input (fake_input would raise).
        self.assertTrue(p.check("bash", {"command": "make build"}))
        # Different command prompts again.
        p2_answers = perms("ask", ["n"])
        p2_answers.allowlist = p.allowlist
        self.assertFalse(p2_answers.check("bash", {"command": "make clean"}))
        # Same path for file writes is remembered, different path is not.
        p3 = perms("ask", ["always"])
        self.assertTrue(p3.check("write_file", {"path": "x.py", "content": "1"}))
        self.assertTrue(p3.check("write_file", {"path": "x.py", "content": "2"}))

    def test_ask_invalid_answer_reprompts(self):
        p = perms("ask", ["maybe", "yes"])
        self.assertTrue(p.check("bash", {"command": "make build"}))

    def test_ask_eof_denies(self):
        def eof(prompt=""):
            raise EOFError
        p = nada.Permissions(mode="ask", input_fn=eof, isatty_fn=lambda: True)
        self.assertFalse(p.check("bash", {"command": "make build"}))

    def test_ask_non_tty_denies_without_prompting(self):
        p = perms("ask", [], tty=False)  # any input() call would raise
        self.assertFalse(p.check("bash", {"command": "make build"}))
        self.assertFalse(p.check("write_file", {"path": "x", "content": "y"}))

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            nada.Permissions(mode="sometimes")

    def test_aliases_normalize_to_canonical_modes(self):
        for alias, canonical in [
            ("ask", "default"), ("default", "default"),
            ("accept-edits", "accept-edits"), ("acceptEdits", "accept-edits"),
            ("approve-edits", "accept-edits"), ("approve_edits", "accept-edits"),
            ("plan", "plan"), ("deny", "plan"), ("dont-ask", "plan"),
            ("auto", "auto"), ("approve-for-me", "auto"), ("auto-review", "auto"),
            ("guardian", "auto"),
            ("yolo", "yolo"), ("bypass-permissions", "yolo"),
        ]:
            self.assertEqual(nada.normalize_mode(alias), canonical, alias)
            self.assertEqual(perms(alias).mode, canonical, alias)
        with self.assertRaises(ValueError):
            nada.normalize_mode("sometimes")

    def test_accept_edits_allows_edits_but_prompts_for_shell(self):
        p = perms("accept-edits")  # no answers: any prompt would raise
        self.assertTrue(p.check("write_file", {"path": "x.py", "content": "1"}))
        self.assertTrue(p.check("edit_file", {"path": "x.py", "old_string": "a", "new_string": "b"}))
        self.assertTrue(p.check("bash", {"command": "ls -R"}))  # read-only still free

        p = perms("approve-edits", ["n"])
        self.assertFalse(p.check("bash", {"command": "make build"}))

        p = perms("acceptEdits", ["a"])
        self.assertTrue(p.check("bash", {"command": "make build"}))
        self.assertTrue(p.check("bash", {"command": "make build"}))  # remembered

    def test_plan_denies_edits_and_shell_without_prompting(self):
        p = perms("plan")  # no answers: any prompt would raise
        self.assertFalse(p.check("write_file", {"path": "x", "content": "y"}))
        self.assertFalse(p.check("edit_file", {"path": "x", "old_string": "a", "new_string": "b"}))
        self.assertFalse(p.check("bash", {"command": "make build"}))
        self.assertTrue(p.check("bash", {"command": "git status"}))
        self.assertTrue(p.check("read_file", {"path": "x"}))

    def test_set_and_cycle_mode(self):
        p = perms("default")
        self.assertEqual(p.set_mode("ask"), "default")
        self.assertEqual([p.cycle_mode() for _ in range(5)],
                         ["accept-edits", "plan", "auto", "yolo", "default"])
        self.assertEqual(p.cycle_mode(-1), "yolo")  # backwards wraps
        self.assertEqual(nada.next_mode("auto"), "yolo")
        self.assertEqual(nada.next_mode("yolo"), "default")
        self.assertEqual(nada.next_mode("default", -1), "yolo")
        with self.assertRaises(ValueError):
            p.set_mode("sometimes")
        self.assertEqual(p.mode, "yolo")  # unchanged on failure

    def test_mode_command(self):
        p = perms("default")
        out = io.StringIO()
        with patch.object(nada, "PERMS", p), patch("sys.stdout", out):
            nada.handle_permissions_command(["/mode", "approve-edits"])
            self.assertEqual(p.mode, "accept-edits")
            nada.handle_permissions_command(["/permissions", "deny"])
            self.assertEqual(p.mode, "plan")
            nada.handle_permissions_command(["/mode", "bogus"])
            self.assertEqual(p.mode, "plan")
            nada.handle_permissions_command(["/permissions", "clear"])
            nada.handle_permissions_command(["/mode"])
        self.assertIn("accept-edits", out.getvalue())


class TestModeSwitchKeys(unittest.TestCase):
    def test_sequences(self):
        for seq in [b"[Z", b"[27;5;9~", b"[27;5;9u", b"[9;5u", b"[1;5I", b"[27;2;9~"]:
            self.assertEqual(nada.mode_switch_direction(seq), 1, seq)
        for seq in [b"[27;6;9~", b"[27;8;9~", b"[1;6I"]:
            self.assertEqual(nada.mode_switch_direction(seq), -1, seq)
        for seq in [b"[A", b"[C", b"[3~", b"[200~", b""]:
            self.assertEqual(nada.mode_switch_direction(seq), 0, seq)

    def _read(self, data, mode="default", history=None):
        p = perms(mode)
        out = io.StringIO()
        line = nada._read_line_raw(p, io.BytesIO(data), out, history)
        return line, p.mode, out.getvalue()

    def test_ctrl_tab_cycles_and_keeps_text(self):
        line, mode, out = self._read(b"\x1b[27;5;9~hello\r")
        self.assertEqual((line, mode), ("hello", "accept-edits"))
        self.assertIn("[accept-edits] > ", out)  # prompt redrawn with new mode

    def test_shift_tab_and_plain_tab_cycle_forward(self):
        self.assertEqual(self._read(b"\x1b[Zx\r")[1], "accept-edits")
        self.assertEqual(self._read(b"\tx\r")[1], "accept-edits")

    def test_ctrl_shift_tab_cycles_backwards(self):
        self.assertEqual(self._read(b"\x1b[27;6;9~x\r")[1], "yolo")

    def test_editing_and_history(self):
        line, _, _ = self._read(b"heX\x7fllo\r")  # backspace fixes typo
        self.assertEqual(line, "hello")
        line, _, _ = self._read(b"\x1b[A\r", history=["old task"])
        self.assertEqual(line, "old task")

    def test_ctrl_c_and_ctrl_d(self):
        p = perms()
        with self.assertRaises(KeyboardInterrupt):
            nada._read_line_raw(p, io.BytesIO(b"\x03"), io.StringIO())
        with self.assertRaises(EOFError):
            nada._read_line_raw(p, io.BytesIO(b"\x04"), io.StringIO())


class TestRunTurns(unittest.TestCase):
    def _run(self, tool_args, p):
        calls = []
        responses = iter([
            {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "1", "function": {"name": "bash", "arguments": tool_args}}]}}]},
            {"choices": [{"message": {"content": "done"}}]},
        ])
        with patch.object(nada, "call_model", lambda messages: next(responses)), \
             patch.object(nada, "PERMS", p), \
             patch.object(nada, "HANDLERS", {"bash": lambda a: calls.append(a) or "ran"}):
            messages = [{"role": "system", "content": ""}]
            nada.run_turns(messages)
        return calls, messages

    def test_denied_tool_is_not_executed(self):
        calls, messages = self._run('{"command": "rm -rf /"}', perms("ask", ["n"]))
        self.assertEqual(calls, [])
        self.assertIn("PermissionDenied", messages[-2]["content"])

    def test_allowed_tool_is_executed(self):
        calls, messages = self._run('{"command": "rm -rf build"}', perms("ask", ["y"]))
        self.assertEqual(calls, [{"command": "rm -rf build"}])
        self.assertEqual(messages[-2]["content"], "ran")


class TestCallModelRetries(unittest.TestCase):
    """429 (rate limit) -> exponential-backoff retries in call_model."""

    @staticmethod
    def _error(code=429, retry_after=None):
        headers = {"Retry-After": retry_after} if retry_after is not None else {}
        return urllib.error.HTTPError(nada.API_URL, code, "error", headers,
                                      io.BytesIO(b"rate limited"))

    class _Response:
        def __init__(self, payload):
            self._buf = io.BytesIO(json.dumps(payload).encode())

        def __enter__(self):
            return self._buf

        def __exit__(self, *exc):
            return False

    def _run(self, outcomes, delays, calls, **kwargs):
        """call_model with urlopen replaying `outcomes` (exceptions raise,
        dicts succeed); sleeps recorded in `delays`, attempts in `calls`."""
        remaining = iter(outcomes)

        def fake_urlopen(req, timeout=None):
            calls.append(req)
            outcome = next(remaining)
            if isinstance(outcome, Exception):
                raise outcome
            return self._Response(outcome)

        kwargs.setdefault("sleep_fn", delays.append)
        with patch.object(nada.urllib.request, "urlopen", fake_urlopen), \
             patch.object(nada, "RETRY_BASE_DELAY", 1.0), \
             patch.object(nada, "RETRY_MAX_DELAY", 60.0), \
             patch.object(nada, "RETRY_BACKOFF", 2.0), \
             patch("sys.stderr", io.StringIO()):
            return nada.call_model([{"role": "user", "content": "hi"}], **kwargs)

    def test_retries_429_with_backoff_then_succeeds(self):
        delays, calls = [], []
        result = self._run([self._error(), self._error(), {"ok": True}], delays, calls)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(calls), 3)  # 1 initial + 2 retries
        self.assertEqual(len(delays), 2)
        # Equal jitter: wait is 50-100% of 1s, then 50-100% of 2s.
        self.assertTrue(0.5 <= delays[0] <= 1.0, delays)
        self.assertTrue(1.0 <= delays[1] <= 2.0, delays)

    def test_honours_retry_after_as_floor(self):
        delays, calls = [], []
        self._run([self._error(retry_after="5"), {"ok": True}], delays, calls)
        self.assertEqual(delays, [5.0])  # never sooner than the server asks

    def test_gives_up_after_max_retries(self):
        delays, calls = [], []
        with self.assertRaises(RuntimeError) as cm:
            self._run([self._error() for _ in range(3)], delays, calls, max_retries=2)
        self.assertIn("429", str(cm.exception))
        self.assertEqual(len(calls), 3)  # 1 initial + 2 retries, then raise
        self.assertEqual(len(delays), 2)

    def test_non_429_is_not_retried(self):
        delays, calls = [], []
        with self.assertRaises(RuntimeError) as cm:
            self._run([self._error(code=500), {"ok": True}], delays, calls)
        self.assertIn("500", str(cm.exception))
        self.assertEqual(len(calls), 1)
        self.assertEqual(delays, [])

    def test_parse_retry_after(self):
        self.assertEqual(nada._parse_retry_after("3"), 3.0)
        self.assertEqual(nada._parse_retry_after("0"), 0.0)
        self.assertIsNone(nada._parse_retry_after(None))
        self.assertIsNone(nada._parse_retry_after(""))
        self.assertIsNone(nada._parse_retry_after("soon"))
        future = formatdate(time.time() + 30, usegmt=True)
        self.assertTrue(25 < nada._parse_retry_after(future) <= 30)
        past = formatdate(time.time() - 30, usegmt=True)
        self.assertEqual(nada._parse_retry_after(past), 0.0)

    def test_backoff_delay_grows_and_caps(self):
        with patch.object(nada, "RETRY_BASE_DELAY", 1.0), \
             patch.object(nada, "RETRY_MAX_DELAY", 60.0), \
             patch.object(nada, "RETRY_BACKOFF", 2.0), \
             patch.object(nada.random, "uniform", lambda a, b: 0):  # no jitter
            self.assertEqual(nada._backoff_delay(0), 0.5)   # 1s nominal, halved
            self.assertEqual(nada._backoff_delay(1), 1.0)   # 2s
            self.assertEqual(nada._backoff_delay(2), 2.0)   # 4s
            self.assertEqual(nada._backoff_delay(100), 30.0)  # capped at 60s nominal
            self.assertEqual(nada._backoff_delay(0, retry_after=10), 10)


class TestModelEffort(unittest.TestCase):
    def setUp(self):
        nada._MODEL_REASONING_CACHE.clear()

    def tearDown(self):
        nada._MODEL_REASONING_CACHE.clear()
        nada.set_effort(None)

    def test_parse_model_reasoning(self):
        info = nada.parse_model_reasoning({"reasoning": {
            "mandatory": True, "supported_efforts": ["max", "xhigh", "high", "medium", "low", "minimal"],
            "default_effort": "max"}})
        self.assertEqual(info["supported_efforts"], ["max", "xhigh", "high", "medium", "low", "minimal"])
        self.assertEqual(info["default_effort"], "max")
        self.assertTrue(info["mandatory"])
        # No reasoning field / no effort selection exposed -> no levels.
        self.assertEqual(nada.parse_model_reasoning({})["supported_efforts"], [])
        self.assertEqual(nada.parse_model_reasoning({"reasoning": None})["supported_efforts"], [])
        self.assertEqual(nada.parse_model_reasoning({"reasoning": {"mandatory": False}})["supported_efforts"], [])
        # Present-but-null supported_efforts -> all gateway levels.
        self.assertEqual(nada.parse_model_reasoning({"reasoning": {"supported_efforts": None}})["supported_efforts"],
                         list(nada.EFFORT_LEVELS))

    def test_normalize_and_validate_effort(self):
        self.assertEqual(nada.normalize_effort("HIGH"), "high")
        self.assertEqual(nada.normalize_effort("off"), "none")
        self.assertIsNone(nada.normalize_effort("default"))
        self.assertIsNone(nada.normalize_effort("clear"))
        self.assertIsNone(nada.normalize_effort(""))
        self.assertIsNone(nada.normalize_effort(None))
        self.assertEqual(nada.validate_effort("low", ["high", "low"]), "low")
        self.assertIsNone(nada.validate_effort("clear", ["high"]))
        with self.assertRaises(ValueError):
            nada.validate_effort("banana")  # not a gateway level at all
        with self.assertRaises(ValueError):
            nada.validate_effort("max", ["high", "low"])  # level the model lacks
        with self.assertRaises(ValueError):
            nada.validate_effort("none", [])  # e.g. mandatory model: no disable
        self.assertEqual(nada.validate_effort("max", None), "max")  # unknown: any level

    def test_set_effort_leaves_unchanged_on_failure(self):
        nada.set_effort("high")
        with self.assertRaises(ValueError):
            nada.set_effort("max", ["high"])
        self.assertEqual(nada.EFFORT, "high")

    def test_discover_efforts_from_openrouter(self):
        seen = []

        class _Response:
            def __init__(self, payload):
                self._buf = io.BytesIO(json.dumps(payload).encode())
            def __enter__(self):
                return self._buf
            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=None):
            seen.append(req.full_url)
            return _Response({"data": {"id": "meta/muse-ember", "reasoning": {
                "mandatory": True, "supported_efforts": ["max", "high", "low"], "default_effort": "max"}}})

        with patch.object(nada.urllib.request, "urlopen", fake_urlopen):
            self.assertEqual(nada.discover_efforts(), ["max", "high", "low"])
            self.assertEqual(nada.discover_efforts(), ["max", "high", "low"])  # cached
        self.assertEqual(len(seen), 1)
        self.assertIn("/api/v1/model/meta/muse-ember", seen[0])

    def test_call_model_sends_reasoning_effort(self):
        bodies = []

        class _Response:
            def __init__(self, payload):
                self._buf = io.BytesIO(json.dumps(payload).encode())
            def __enter__(self):
                return self._buf
            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=None):
            bodies.append(json.loads(req.data))
            return _Response({"ok": True})

        with patch.object(nada.urllib.request, "urlopen", fake_urlopen):
            with patch.object(nada, "EFFORT", "high"):
                nada.call_model([{"role": "user", "content": "hi"}])
                nada.call_model([{"role": "user", "content": "hi"}], model="other/model")
            with patch.object(nada, "EFFORT", None):
                nada.call_model([{"role": "user", "content": "hi"}])
        self.assertEqual(bodies[0]["reasoning"], {"effort": "high"})       # main model
        self.assertNotIn("reasoning", bodies[1])                            # other (reviewer) model
        self.assertNotIn("reasoning", bodies[2])                            # unset = model default

    def test_effort_command(self):
        out = io.StringIO()
        with patch.object(nada, "discover_efforts", lambda: ["max", "high", "low"]), \
             patch.object(nada, "fetch_model_reasoning", lambda: nada.parse_model_reasoning(
                 {"reasoning": {"supported_efforts": ["max", "high", "low"], "default_effort": "max"}})), \
             patch("sys.stdout", out):
            nada.handle_effort_command(["/effort"])
            nada.handle_effort_command(["/effort", "HIGH"])
            self.assertEqual(nada.EFFORT, "high")
            nada.handle_effort_command(["/effort", "minimal"])  # not offered by this model
            self.assertEqual(nada.EFFORT, "high")               # unchanged
            nada.handle_effort_command(["/effort", "clear"])
            self.assertIsNone(nada.EFFORT)
        text = out.getvalue()
        self.assertIn("max", text)
        self.assertIn("Effort set: high", text)
        self.assertIn("Invalid effort", text)


class TestParseArgs(unittest.TestCase):
    def test_task_and_flags(self):
        a = nada.parse_args(["--yes", "do", "the", "thing"])
        self.assertTrue(a.yes)
        self.assertEqual(a.task, ["do", "the", "thing"])
        self.assertIsNone(a.permission_mode)

        a = nada.parse_args(["--permission-mode", "deny"])
        self.assertEqual(a.permission_mode, "deny")
        self.assertEqual(a.task, [])

        a = nada.parse_args(["--mode", "approve-edits"])
        self.assertEqual(a.permission_mode, "approve-edits")

        a = nada.parse_args(["--accept-edits", "do", "it"])
        self.assertTrue(a.accept_edits)
        self.assertFalse(a.plan)
        self.assertEqual(a.task, ["do", "it"])

        a = nada.parse_args(["--plan"])
        self.assertTrue(a.plan)

        a = nada.parse_args(["hello"])
        self.assertFalse(a.yes)
        self.assertEqual(a.task, ["hello"])
        self.assertIsNone(a.effort)

        a = nada.parse_args(["--effort", "high", "do", "it"])
        self.assertEqual(a.effort, "high")
        self.assertEqual(a.task, ["do", "it"])

        a = nada.parse_args(["--reasoning-effort", "low"])
        self.assertEqual(a.effort, "low")


if __name__ == "__main__":
    unittest.main()
