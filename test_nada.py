#!/usr/bin/env python3
import io
import json
import os
import tempfile
import threading
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
        with tempfile.TemporaryDirectory() as tmp:
            settings = os.path.join(tmp, "settings.json")
            with patch.dict(os.environ, {"NADA_SETTINGS_FILE": settings}), \
                 patch.object(nada, "PERMS", p), patch("sys.stdout", out):
                nada.handle_permissions_command(["/mode", "approve-edits"])
                self.assertEqual(p.mode, "accept-edits")
                self.assertEqual(nada.load_settings(settings)["permission_mode"],
                                 "accept-edits")  # saved for next run
                nada.handle_permissions_command(["/permissions", "deny"])
                self.assertEqual(p.mode, "plan")
                self.assertEqual(nada.load_settings(settings)["permission_mode"], "plan")
                nada.handle_permissions_command(["/mode", "bogus"])
                self.assertEqual(p.mode, "plan")  # unchanged, not re-saved
                self.assertEqual(nada.load_settings(settings)["permission_mode"], "plan")
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
        with tempfile.TemporaryDirectory() as tmp:  # keep cycle-persistence off the real file
            with patch.dict(os.environ, {"NADA_SETTINGS_FILE": os.path.join(tmp, "settings.json")}):
                line = nada._read_line_raw(p, io.BytesIO(data), out, history)
        return line, p.mode, out.getvalue()

    def test_ctrl_tab_cycles_and_keeps_text(self):
        line, mode, out = self._read(b"\x1b[27;5;9~hello\r")
        self.assertEqual((line, mode), ("hello", "accept-edits"))
        self.assertIn("[accept-edits] > ", out)  # prompt redrawn with new mode

    def test_ctrl_tab_cycle_saves_mode_for_next_run(self):
        p = perms("default")
        with tempfile.TemporaryDirectory() as tmp:
            settings = os.path.join(tmp, "settings.json")
            with patch.dict(os.environ, {"NADA_SETTINGS_FILE": settings}):
                nada._read_line_raw(p, io.BytesIO(b"\x1b[27;6;9~x\r"), io.StringIO())
            self.assertEqual(nada.load_settings(settings)["permission_mode"], "yolo")

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


class TestReviewContextV1(unittest.TestCase):
    """Auto-review v1: full conversation + retained user requests, exact
    action, score-derived verdicts, denial resample, rationale surfacing."""

    def test_user_requests_always_retained(self):
        # The original task must survive no matter how many tool results
        # follow it (the issue #1 failure: last-6 + head-truncation lost it).
        messages = [{"role": "system", "content": "scaffold"},
                    {"role": "user", "content": "add AGENTS.md support"}]
        for i in range(20):
            messages += [{"role": "assistant", "content": "reading"},
                         {"role": "tool", "tool_call_id": str(i), "content": "y" * 2000}]
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": "9", "function": {"name": "bash",
                                            "arguments": '{"command":"make build"}'}}]})
        self.assertEqual(nada.collect_user_requests(messages), ["add AGENTS.md support"])
        ctx = nada.build_review_context(messages)
        self.assertIn("add AGENTS.md support", ctx)          # task retained
        self.assertIn("make build", ctx)                      # tail not cut
        self.assertNotIn("scaffold", ctx.split("CONVERSATION")[1])  # system excluded

    def test_build_review_context_string_passthrough(self):
        self.assertEqual(nada.build_review_context("plain"), "plain")
        self.assertEqual(nada.build_review_context(None), "")

    def test_action_json_is_untruncated(self):
        args = {"path": "nada.py", "old_string": "a", "new_string": "x" * 5000 + "TAIL"}
        self.assertIn("TAIL", nada._action_json("edit_file", args))

    def test_parse_verdict_whitespace_and_scores(self):
        self.assertTrue(nada._parse_review_verdict('{"decision":" approve ","rationale":"x"}')[0])
        # No decision: derived from Codex thresholds.
        self.assertTrue(nada._parse_review_verdict(
            '{"risk_level":"low","user_authorization":"unknown","rationale":"r"}')[0])
        self.assertTrue(nada._parse_review_verdict(
            '{"risk_level":"high","user_authorization":"high","rationale":"r"}')[0])
        self.assertFalse(nada._parse_review_verdict(
            '{"risk_level":"high","user_authorization":"unknown","rationale":"r"}')[0])
        self.assertFalse(nada._parse_review_verdict(
            '{"risk_level":"critical","user_authorization":"high","rationale":"r"}')[0])
        # Provider content as a list of parts.
        self.assertTrue(nada._parse_review_verdict(
            [{"text": '{"decision":"approve","rationale":"ok"}'}])[0])

    def _review(self, replies, name="edit_file", args=None, context=None):
        """review_action with call_model replaying `replies` (dicts or
        exceptions); returns (verdict, sent_messages)."""
        sent, remaining = [], iter(replies)

        def fake_call_model(messages, **kw):
            sent.append(messages)
            outcome = next(remaining)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch.object(nada, "call_model", fake_call_model):
            return nada.review_action(name, args or {"path": "x"}, context), sent

    @staticmethod
    def _reply(content):
        return {"choices": [{"message": {"content": content}}]}

    def test_review_action_sends_full_action_and_context(self):
        big = {"path": "x", "content": "z" * 3000 + "TAIL"}
        messages = [{"role": "user", "content": "do the thing"},
                    {"role": "assistant", "content": "working"}]
        (approved, _), sent = self._review(
            [self._reply('{"decision":"approve","rationale":"fine"}')],
            name="write_file", args=big, context=messages)
        self.assertTrue(approved)
        user_msg = sent[0][1]["content"]
        self.assertIn("TAIL", user_msg)             # exact action, untruncated
        self.assertIn("do the thing", user_msg)     # user request retained
        # Stable context precedes the action, so the whole context is a
        # cacheable prefix across successive reviews in a session.
        self.assertLess(user_msg.index("do the thing"), user_msg.index("TAIL"))

    def test_review_action_resamples_a_denial_once(self):
        deny = self._reply('{"decision":"deny","rationale":"unsure"}')
        allow = self._reply('{"decision":"approve","rationale":"fine on resample"}')
        (approved, rationale), sent = self._review([deny, allow])
        self.assertTrue(approved)
        self.assertEqual(len(sent), 2)

    def test_review_action_failed_resample_keeps_first_denial(self):
        deny = self._reply('{"decision":"deny","rationale":"first"}')
        (approved, rationale), _ = self._review([deny, RuntimeError("boom")])
        self.assertFalse(approved)
        self.assertEqual(rationale, "first")

    def test_review_action_retries_reviewer_error_once(self):
        allow = self._reply('{"decision":"approve","rationale":"ok"}')
        (approved, _), _ = self._review([RuntimeError("flake"), allow])
        self.assertTrue(approved)
        with self.assertRaises(RuntimeError):
            self._review([RuntimeError("x"), RuntimeError("y")])

    def test_review_action_tolerates_list_content_and_empty_choices(self):
        (approved, _), _ = self._review(
            [self._reply([{"text": '{"decision":"approve","rationale":"ok"}'}])])
        self.assertTrue(approved)
        (approved, _), _ = self._review([{"choices": []}, {"choices": []}])
        self.assertFalse(approved)  # no verdict -> deny (fail closed)

    def test_call_model_json_mode(self):
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
            nada.call_model([{"role": "user", "content": "hi"}], json_mode=True)
            nada.call_model([{"role": "user", "content": "hi"}])
        self.assertEqual(bodies[0]["response_format"], {"type": "json_object"})
        self.assertNotIn("response_format", bodies[1])


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

    def test_auto_denial_rationale_reaches_the_model(self):
        # Issue #1: the agent must see WHY it was denied, not a bare
        # PermissionDenied it cannot act on.
        def reviewer(name, args, context=None):
            return (False, "would weaken security boundaries")
        calls, messages = self._run('{"command": "rm -rf build"}',
                                    perms("auto", reviewer=reviewer))
        self.assertEqual(calls, [])
        self.assertIn("would weaken security boundaries", messages[-2]["content"])

    def test_auto_reviewer_error_is_distinguishable(self):
        def reviewer(name, args, context=None):
            raise RuntimeError("model gone")
        calls, messages = self._run('{"command": "make build"}',
                                    perms("auto", reviewer=reviewer))
        self.assertEqual(calls, [])
        self.assertIn("auto-reviewer unavailable: model gone", messages[-2]["content"])


class TestEscInterrupt(unittest.TestCase):
    """Esc (or Ctrl-C) during a turn interrupts it and returns to the prompt."""

    def tearDown(self):
        nada._TURN_INTERRUPTED.clear()

    def _listen(self, payloads):
        """Run _listen_for_esc on a pipe, feeding `payloads`; returns
        (interrupted.is_set(), [(pid, sig), ...])."""
        r, w = os.pipe()
        stop, interrupted = threading.Event(), threading.Event()
        paused, ack = threading.Event(), threading.Event()
        kills = []
        t = threading.Thread(target=nada._listen_for_esc,
                             args=(r, stop, interrupted, paused, ack),
                             daemon=True)
        t.start()
        try:
            with patch.object(nada.os, "kill",
                             lambda pid, sig: kills.append((pid, sig))):
                for p in payloads:
                    os.write(w, p)
                    time.sleep(0.15)  # let the listener drain each payload
                t.join(timeout=2)
        finally:
            stop.set()
            t.join(timeout=2)
            os.close(r)
            os.close(w)
        return interrupted.is_set(), kills

    def test_lone_esc_sets_interrupt_and_signals(self):
        interrupted, kills = self._listen([b"\x1b"])
        self.assertTrue(interrupted)
        self.assertEqual(kills, [(os.getpid(), nada.signal.SIGINT)])

    def test_escape_sequence_and_alt_key_are_not_an_interrupt(self):
        # Up-arrow, then Alt+x, then a real lone Esc (which must still work,
        # proving the earlier bytes were drained, not taken for an Esc).
        interrupted, kills = self._listen([b"\x1b[A", b"\x1bx", b"\x1b"])
        self.assertTrue(interrupted)
        self.assertEqual(kills, [(os.getpid(), nada.signal.SIGINT)])

    def test_other_keys_are_not_an_interrupt(self):
        interrupted, kills = self._listen([b"hello\r"])
        self.assertFalse(interrupted)
        self.assertEqual(kills, [])

    def test_run_turns_returns_to_prompt_when_model_is_interrupted(self):
        def boom(messages):
            raise KeyboardInterrupt
        out = io.StringIO()
        messages = [{"role": "system", "content": ""},
                    {"role": "user", "content": "hi"}]
        with patch.object(nada, "call_model", boom), patch("sys.stdout", out):
            nada.run_turns(messages)  # must not raise
        self.assertIn("Interrupted", out.getvalue())
        self.assertEqual(messages[-1], {"role": "user", "content": "hi"})

    def test_run_turns_trims_dangling_tool_calls_on_interrupt(self):
        responses = iter([
            {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
                {"id": "1", "function": {"name": "bash",
                                          "arguments": '{"command": "make"}'}}]}}]},
        ])
        def boom(args):
            raise KeyboardInterrupt
        messages = [{"role": "system", "content": ""},
                    {"role": "user", "content": "hi"}]
        with patch.object(nada, "call_model", lambda m: next(responses)), \
             patch.object(nada, "PERMS", perms("yolo")), \
             patch.object(nada, "HANDLERS", {"bash": boom}), \
             patch("sys.stdout", io.StringIO()):
            nada.run_turns(messages)  # must not raise
        # The assistant tool_calls message had no result: dropped, so the
        # next API call gets no dangling tool_calls.
        self.assertEqual(messages, [{"role": "system", "content": ""},
                                    {"role": "user", "content": "hi"}])

    def test_run_turns_aborts_before_the_next_tool_call(self):
        calls = []
        responses = iter([
            {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "1", "function": {"name": "bash", "arguments": "{}"}},
                {"id": "2", "function": {"name": "bash", "arguments": "{}"}}]}}]},
        ])
        def first(args):
            calls.append(args)
            nada._TURN_INTERRUPTED.set()  # as if Esc was pressed mid-tool
            return "ran"
        messages = [{"role": "system", "content": ""},
                    {"role": "user", "content": "hi"}]
        with patch.object(nada, "call_model", lambda m: next(responses)), \
             patch.object(nada, "PERMS", perms("yolo")), \
             patch.object(nada, "HANDLERS", {"bash": first}), \
             patch("sys.stdout", io.StringIO()):
            nada.run_turns(messages)
        self.assertEqual(len(calls), 1)  # second tool call never ran

    def test_permission_prompt_lone_esc_interrupts(self):
        with self.assertRaises(nada.TurnInterrupted):
            perms("ask", ["\x1b"]).check("bash", {"command": "make build"})

    def test_suspend_resume_without_a_listener_is_a_noop(self):
        self.assertIsNone(nada._ACTIVE_ESC_LISTENER)
        nada._suspend_esc_listener()
        nada._resume_esc_listener()


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

    def test_set_model(self):
        original = nada.MODEL
        self.addCleanup(lambda: nada.set_model(original))
        self.assertEqual(nada.set_model("  openai/gpt-5.2 "), "openai/gpt-5.2")
        for bad in ["", "   ", "a b", None, 42]:
            with self.assertRaises(ValueError, msg=repr(bad)):
                nada.set_model(bad)
        self.assertEqual(nada.MODEL, "openai/gpt-5.2")  # unchanged on failure

    def test_model_command(self):
        original = nada.MODEL
        self.addCleanup(lambda: nada.set_model(original))
        with tempfile.TemporaryDirectory() as tmp:
            settings = os.path.join(tmp, "nested", "settings.json")
            out = io.StringIO()
            with patch.dict(os.environ, {"NADA_SETTINGS_FILE": settings}), \
                 patch.object(nada, "discover_efforts", lambda model=None: ["high", "low"]), \
                 patch("sys.stdout", out):
                nada.handle_model_command(["/model"])
                self.assertIn(f"Model: {original}", out.getvalue())
                nada.set_effort("high")
                nada.handle_model_command(["/model", "openai/gpt-5.2"])
                self.assertEqual(nada.MODEL, "openai/gpt-5.2")
                self.assertEqual(nada.EFFORT, "high")  # offered by new model: kept
                saved = nada.load_settings(settings)
                self.assertEqual((saved["model"], saved["effort"]),
                                 ("openai/gpt-5.2", "high"))  # saved
                nada.set_effort("max")
                nada.handle_model_command(["/model", "other/model"])
                self.assertIsNone(nada.EFFORT)  # not offered by other/model: cleared
                nada.handle_model_command(["/model", "not a model"])
                self.assertEqual(nada.MODEL, "other/model")  # unchanged on failure
                nada.handle_model_command(["/model", "a", "b"])
                self.assertIn("Usage: /model", out.getvalue())
            text = out.getvalue()
            self.assertIn("Model set: openai/gpt-5.2", text)
            self.assertIn("not supported", text)

    def test_model_command_keeps_effort_when_discovery_fails(self):
        original = nada.MODEL
        self.addCleanup(lambda: nada.set_model(original))

        def boom(model=None):
            raise RuntimeError("offline")
        with patch.object(nada, "discover_efforts", boom), \
             patch.object(nada, "persist_settings", lambda path=None: True), \
             patch("sys.stdout", io.StringIO()):
            nada.set_effort("high")
            nada.handle_model_command(["/model", "other/model"])
        self.assertEqual(nada.EFFORT, "high")  # kept: OpenRouter judges

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

    def test_detect_llama_app_and_use_its_openai_endpoint(self):
        seen, bodies = [], []

        class _Response:
            def __init__(self, payload):
                self._buf = io.BytesIO(json.dumps(payload).encode())
            def __enter__(self):
                return self._buf
            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=None):
            seen.append(req)
            if req.full_url.endswith("/v1/models"):
                return _Response({"data": [{"id": "local-coder"}]})
            bodies.append(json.loads(req.data))
            return _Response({"ok": True})

        with patch.object(nada.urllib.request, "urlopen", fake_urlopen), \
             patch.object(nada, "LLAMA_APP_URL", "http://127.0.0.1:9931"), \
             patch.object(nada, "LLAMA_APP_API_KEY", ""), \
             patch.object(nada, "USING_LLAMA_APP", False), \
             patch.object(nada, "API_URL", "https://openrouter.ai/api/v1/chat/completions"):
            self.assertEqual(nada.detect_llama_app(), ["local-coder"])
            nada.activate_llama_app()
            nada.call_model([{"role": "user", "content": "hi"}])
            self.assertEqual(nada.API_URL, "http://127.0.0.1:9931/v1/chat/completions")
            self.assertNotIn("Authorization", seen[-1].headers)
        self.assertEqual(bodies[0]["model"], nada.MODEL)

    def test_detect_llama_app_falls_back_on_connection_error(self):
        with patch.object(nada.urllib.request, "urlopen", side_effect=urllib.error.URLError("offline")):
            self.assertEqual(nada.detect_llama_app(), [])

    def test_effort_command(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            settings = os.path.join(tmp, "settings.json")
            with patch.dict(os.environ, {"NADA_SETTINGS_FILE": settings}), \
                 patch.object(nada, "discover_efforts", lambda: ["max", "high", "low"]), \
                 patch.object(nada, "fetch_model_reasoning", lambda: nada.parse_model_reasoning(
                     {"reasoning": {"supported_efforts": ["max", "high", "low"], "default_effort": "max"}})), \
                 patch("sys.stdout", out):
                nada.handle_effort_command(["/effort"])
                nada.handle_effort_command(["/effort", "HIGH"])
                self.assertEqual(nada.EFFORT, "high")
                self.assertEqual(nada.load_settings()["effort"], "high")  # saved
                nada.handle_effort_command(["/effort", "minimal"])  # not offered by this model
                self.assertEqual(nada.EFFORT, "high")               # unchanged
                nada.handle_effort_command(["/effort", "clear"])
                self.assertIsNone(nada.EFFORT)
                self.assertIsNone(nada.load_settings()["effort"])  # clear is saved too
        text = out.getvalue()
        self.assertIn("max", text)
        self.assertIn("Effort set: high", text)
        self.assertIn("Invalid effort", text)


class TestPersistentSettings(unittest.TestCase):
    """~/.nada/settings.json: the last model/effort/permission mode
    selected in nada is reloaded on the next run (CLI > env > saved >
    default)."""

    def test_default_settings_file_env(self):
        with patch.dict(os.environ, {"NADA_SETTINGS_FILE": "/tmp/my-settings.json"}):
            self.assertEqual(str(nada.default_settings_file()), "/tmp/my-settings.json")

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "settings.json")  # parents created
            nada.save_settings({"model": "openai/gpt-5.2", "effort": "HIGH",
                                "permission_mode": "approve-edits"}, path=path)
            self.assertEqual(nada.load_settings(path),
                             {"model": "openai/gpt-5.2", "effort": "high",
                              "permission_mode": "accept-edits"})  # alias canonicalized
            with open(path) as f:
                raw = json.load(f)
            self.assertEqual(raw["version"], nada.SETTINGS_VERSION)
            self.assertEqual(os.listdir(os.path.join(tmp, "nested")),
                             ["settings.json"])  # no tmp left

    def test_load_missing_or_corrupt_is_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(nada.load_settings(os.path.join(tmp, "absent.json")), {})
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w") as f:
                f.write("not json")
            self.assertEqual(nada.load_settings(bad), {})
            with open(bad, "w") as f:
                f.write('{"version": 999, "model": "x/y"}')
            self.assertEqual(nada.load_settings(bad), {})

    def test_load_drops_invalid_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "settings.json")
            with open(path, "w") as f:
                json.dump({"version": 1, "model": "not a model", "effort": "banana",
                           "permission_mode": "sometimes"}, f)
            self.assertEqual(nada.load_settings(path),
                             {"model": None, "effort": None, "permission_mode": None})

    def test_persist_settings_saves_current_model_and_effort(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "settings.json")
            with patch.object(nada, "MODEL", "openai/gpt-5.2"), \
                 patch.object(nada, "EFFORT", "low"), \
                 patch.object(nada, "PERMS", perms("yolo")):
                self.assertTrue(nada.persist_settings(path))
            self.assertEqual(nada.load_settings(path),
                             {"model": "openai/gpt-5.2", "effort": "low",
                              "permission_mode": "yolo"})
            # A failure (path is a directory) warns and returns False, no raise.
            with patch("sys.stderr", io.StringIO()):
                self.assertFalse(nada.persist_settings(tmp))

    def test_save_settings_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "settings.json")
            with self.assertRaises(ValueError):
                nada.save_settings({"model": "a b"}, path=path)
            with self.assertRaises(ValueError):
                nada.save_settings({"effort": "banana"}, path=path)
            with self.assertRaises(ValueError):
                nada.save_settings({"permission_mode": "sometimes"}, path=path)
            self.assertFalse(os.path.exists(path))

    def test_resolve_selection_precedence(self):
        saved = {"model": "saved/model", "effort": "low"}
        clean = {k: v for k, v in os.environ.items()
                 if k not in {"MODEL", *nada._EFFORT_ENV_VARS}}
        with patch.dict(os.environ, clean, clear=True), \
             patch.object(nada, "MODEL", "default/model"), \
             patch.object(nada, "EFFORT", None):
            # saved beats the built-in default
            self.assertEqual(nada.resolve_selection(None, None, saved),
                             ("saved/model", "low", "saved", "saved"))
            # CLI beats saved
            self.assertEqual(nada.resolve_selection("cli/model", "high", saved),
                             ("cli/model", "high", "cli", "cli"))
            # nothing saved: built-in defaults
            self.assertEqual(nada.resolve_selection(None, None, {}),
                             ("default/model", None, "default", "default"))
        with patch.dict(os.environ, {"MODEL": "env/model", "AGENT_EFFORT": "max"}), \
             patch.object(nada, "MODEL", "env/model"), \
             patch.object(nada, "EFFORT", "max"):
            # env beats saved, but CLI still beats env
            self.assertEqual(nada.resolve_selection(None, None, saved),
                             ("env/model", "max", "env", "env"))
            self.assertEqual(nada.resolve_selection("cli/model", None, saved),
                             ("cli/model", "max", "cli", "env"))

    def test_resolve_permission_mode_precedence(self):
        saved = {"permission_mode": "plan"}
        p = perms("default")
        with patch.dict(os.environ, {}, clear=True), patch.object(nada, "PERMS", p):
            self.assertEqual(nada.resolve_permission_mode(None, saved), ("plan", "saved"))
            self.assertEqual(nada.resolve_permission_mode("yolo", saved), ("yolo", "cli"))
            self.assertEqual(nada.resolve_permission_mode(None, {}), ("default", "default"))
        p = perms("auto")
        with patch.dict(os.environ, {"AGENT_PERMISSION_MODE": "auto"}), patch.object(nada, "PERMS", p):
            self.assertEqual(nada.resolve_permission_mode(None, saved), ("auto", "env"))
            self.assertEqual(nada.resolve_permission_mode("plan", saved), ("plan", "cli"))

    def test_main_loads_saved_settings_and_persists_cli_selection(self):
        original_model, original_effort = nada.MODEL, nada.EFFORT
        self.addCleanup(lambda: (nada.set_model(original_model),
                                 nada.set_effort(original_effort)))
        clean = {k: v for k, v in os.environ.items()
                 if k not in {"MODEL", "AGENT_PERMISSION_MODE", *nada._EFFORT_ENV_VARS}}
        reasoning = nada.parse_model_reasoning(
            {"reasoning": {"supported_efforts": ["high", "low"]}})
        with tempfile.TemporaryDirectory() as tmp:
            settings = os.path.join(tmp, "settings.json")
            nada.save_settings({"model": "saved/model", "effort": "low",
                                "permission_mode": "plan"}, path=settings)
            env = {**clean, "NADA_SETTINGS_FILE": settings}
            p = perms("default")
            with patch.dict(os.environ, env, clear=True), \
                 patch.object(nada, "API_KEY", "test-key"), \
                 patch.object(nada, "PERMS", p), \
                 patch.object(nada, "fetch_model_reasoning", lambda: reasoning), \
                 patch.object(nada, "run_turns", lambda messages: None), \
                 patch("sys.argv",
                       ["nada.py", "--session-dir", os.path.join(tmp, "sessions"), "hi"]):
                nada.main()  # no CLI selection: saved settings are loaded
            self.assertEqual((nada.MODEL, nada.EFFORT, p.mode), ("saved/model", "low", "plan"))
            with patch.dict(os.environ, env, clear=True), \
                 patch.object(nada, "API_KEY", "test-key"), \
                 patch.object(nada, "PERMS", p), \
                 patch.object(nada, "fetch_model_reasoning", lambda: reasoning), \
                 patch.object(nada, "run_turns", lambda messages: None), \
                 patch("sys.argv",
                       ["nada.py", "--model", "cli/model", "--effort", "high", "--accept-edits",
                        "--session-dir", os.path.join(tmp, "sessions"), "hi"]):
                nada.main()  # CLI selection wins and is saved for next time
            self.assertEqual((nada.MODEL, nada.EFFORT, p.mode), ("cli/model", "high", "accept-edits"))
            self.assertEqual(nada.load_settings(settings),
                             {"model": "cli/model", "effort": "high",
                              "permission_mode": "accept-edits"})

    def test_main_drops_unsupported_saved_effort_and_remembers(self):
        original_model, original_effort = nada.MODEL, nada.EFFORT
        self.addCleanup(lambda: (nada.set_model(original_model),
                                 nada.set_effort(original_effort)))
        clean = {k: v for k, v in os.environ.items()
                 if k not in {"MODEL", *nada._EFFORT_ENV_VARS}}
        reasoning = nada.parse_model_reasoning(
            {"reasoning": {"supported_efforts": ["low"]}})  # saved "max" not offered
        with tempfile.TemporaryDirectory() as tmp:
            settings = os.path.join(tmp, "settings.json")
            nada.save_settings({"model": "saved/model", "effort": "max"}, path=settings)
            with patch.dict(os.environ, {**clean, "NADA_SETTINGS_FILE": settings}, clear=True), \
                 patch.object(nada, "API_KEY", "test-key"), \
                 patch.object(nada, "fetch_model_reasoning", lambda: reasoning), \
                 patch.object(nada, "run_turns", lambda messages: None), \
                 patch("sys.argv",
                       ["nada.py", "--session-dir", os.path.join(tmp, "sessions"), "hi"]), \
                 patch("sys.stderr", io.StringIO()):
                nada.main()  # must not exit: a remembered effort is fail-soft
            self.assertEqual((nada.MODEL, nada.EFFORT), ("saved/model", None))
            self.assertIsNone(nada.load_settings(settings)["effort"])  # drop remembered

    def test_main_uses_detected_llama_app_without_openrouter_key(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items()
                   if k not in {"MODEL", *nada._EFFORT_ENV_VARS}}
            env["NADA_SETTINGS_FILE"] = os.path.join(tmp, "settings.json")
            with patch.dict(os.environ, env, clear=True), \
                 patch.object(nada, "API_KEY", ""), \
                 patch.object(nada, "MODEL", "meta/muse-ember"), \
                 patch.object(nada, "EFFORT", None), \
                 patch.object(nada, "detect_llama_app", lambda: ["local-coder"]), \
                 patch.object(nada, "run_turns", lambda messages: seen.append(
                     (nada.MODEL, nada.API_URL, messages[-1]["content"]))), \
                 patch("sys.argv", ["nada.py", "--session-dir", os.path.join(tmp, "sessions"), "hi"]):
                nada.main()
        self.assertEqual(seen, [("local-coder", "http://127.0.0.1:9931/v1/chat/completions", "hi")])


class TestSessions(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = nada.SessionStore(os.path.join(self._tmp.name, "sessions"))
        patcher = patch.object(nada, "PERMS", perms("default"))
        patcher.start()
        self.addCleanup(patcher.stop)
        nada.set_effort(None)
        self.addCleanup(lambda: nada.set_effort(None))

    def _session(self, first="do the thing"):
        s = nada.Session(self.store)
        s.messages.append({"role": "user", "content": first})
        s.messages.append({"role": "assistant", "content": "done"})
        s.history.append(first)
        return s

    def test_sanitize_rejects_paths(self):
        for bad in ["../x", "a/b", "a\\b", "", ".hidden", "a.json", "a b", ".."]:
            with self.assertRaises(ValueError, msg=bad):
                nada.sanitize_session_id(bad)
        self.assertEqual(nada.sanitize_session_id("20260101-120000-abcd1234"), "20260101-120000-abcd1234")

    def test_validate_rejects_bad_data(self):
        good = {"version": 1, "id": "abc123",
                "messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(nada._validate_session_data(good)["id"], "abc123")
        for bad in [{**good, "version": 999}, {**good, "messages": []},
                    {**good, "messages": [{"content": "x"}]},
                    {**good, "allowlist": [["bash"]]}, {**good, "history": [1]}]:
            with self.assertRaises(ValueError, msg=repr(bad)):
                nada._validate_session_data(bad)

    def test_trim_and_title(self):
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "  fix   the\nbug  "},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {}}]}]
        self.assertEqual(nada._derive_title(msgs), "fix the")
        self.assertEqual(len(nada._trim_incomplete_tool_calls(msgs)), 2)  # dangling call dropped
        complete = msgs + [{"role": "tool", "tool_call_id": "1", "content": "ok"}]
        self.assertEqual(len(nada._trim_incomplete_tool_calls(complete)), 4)

    def test_default_session_dir_env(self):
        with patch.dict(os.environ, {"AGENT_SESSION_DIR": "/tmp/my-sessions"}):
            self.assertEqual(str(nada.default_session_dir()), "/tmp/my-sessions")

    def test_roundtrip_preserves_tool_calls(self):
        s = self._session()
        s.messages[-1] = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]}
        s.messages.append({"role": "tool", "tool_call_id": "1", "content": "ran"})
        s.messages.append({"role": "assistant", "content": "done"})
        path = s.save()
        self.assertTrue(path.is_file())
        self.assertEqual(os.listdir(self._tmp.name + "/sessions"), [f"{s.id}.json"])  # no tmp left
        loaded = self.store.load(s.id)
        self.assertEqual(loaded["messages"], s.messages)
        self.assertEqual(loaded["history"], [s.history[0]])
        self.assertEqual(loaded["title"], "do the thing")

    def test_list_latest_delete_and_corrupt_skipped(self):
        s1, s2 = self._session("first"), self._session("second")
        s1.save(); s2.save()
        summaries = self.store.list()
        self.assertEqual([x["id"] for x in summaries], [s2.id, s1.id])  # newest first
        self.assertEqual(self.store.load("latest")["id"], s2.id)
        with open(os.path.join(self._tmp.name, "sessions", "bad.json"), "w") as f:
            f.write("not json")
        self.assertEqual(len(self.store.list()), 2)  # corrupt file skipped
        self.assertTrue(self.store.delete(s1.id))
        self.assertFalse(self.store.delete(s1.id))
        with self.assertRaises(FileNotFoundError):
            self.store.load(s1.id)

    def test_try_save_only_with_a_turn(self):
        s = nada.Session(self.store)
        self.assertFalse(s.try_save())  # nothing to save yet
        self.assertFalse(self.store.dir.exists())
        s.messages.append({"role": "user", "content": "hi"})
        self.assertTrue(s.try_save())
        s.autosave = False
        self.assertFalse(s.try_save())

    def test_restore_keeps_current_permissions(self):
        nada.PERMS.set_mode("accept-edits")
        nada.PERMS.allowlist.add(("bash", "make build"))
        nada.set_effort("high")
        s = self._session()
        data = s.snapshot()
        self.assertEqual(data["permission_mode"], "accept-edits")  # recorded...
        self.assertEqual(data["allowlist"], [["bash", "make build"]])
        s.save()
        # Change live state, then restore: messages/effort come back,
        # but a file must not silently escalate permissions.
        nada.PERMS.set_mode("plan")
        nada.PERMS.allowlist.clear()
        nada.set_effort("low")
        restored = nada.Session(self.store)
        with patch("sys.stdout", io.StringIO()):
            restored.restore(self.store.load(s.id))
        self.assertEqual(restored.messages[1:], s.messages[1:])
        self.assertEqual(restored.messages[0]["content"], nada.SYSTEM)  # regenerated
        self.assertEqual(restored.history, s.history)
        self.assertEqual(nada.EFFORT, "high")
        self.assertEqual(nada.PERMS.mode, "plan")       # ...not restored
        self.assertEqual(nada.PERMS.allowlist, set())    # ...not restored

    def test_restore_applies_model_and_snapshot_uses_live_model(self):
        original = nada.MODEL
        self.addCleanup(lambda: nada.set_model(original))
        nada.set_model("other/model")
        s = self._session()
        s.save()
        self.assertEqual(self.store.load(s.id)["model"], "other/model")
        nada.set_model("third/model")
        restored = nada.Session(self.store)
        with patch("sys.stdout", io.StringIO()):
            restored.restore(self.store.load(s.id))
        self.assertEqual(nada.MODEL, "other/model")  # saved model is applied
        self.assertEqual(restored.model, "other/model")
        nada.set_model("fourth/model")  # /model after restore...
        self.assertEqual(restored.snapshot()["model"], "fourth/model")  # ...is what saves
        with patch("sys.stdout", io.StringIO()):
            restored.restore(self.store.load(s.id), apply_model=False)
        self.assertEqual(nada.MODEL, "fourth/model")  # CLI --model wins over the session

    def test_session_command(self):
        s = self._session()
        s.save()
        other = nada.Session(self.store)
        out = io.StringIO()
        settings = os.path.join(self._tmp.name, "settings.json")
        with patch.dict(os.environ, {"NADA_SETTINGS_FILE": settings}), patch("sys.stdout", out):
            nada.handle_session_command(["/session", "list"], other)
            nada.handle_session_command(["/resume", s.id], other)
            nada.handle_session_command(["/session"], other)
            nada.handle_session_command(["/session", "resume", "missing"], other)
        # Resuming saved the session's model/effort as the next-run defaults.
        self.assertEqual(nada.load_settings(settings)["model"], nada.MODEL)
        self.assertEqual(other.turns, 1)  # resumed before the delete below
        with patch("sys.stdout", out):
            nada.handle_session_command(["/session", "delete", s.id], other)
        text = out.getvalue()
        self.assertIn(s.id, text)
        self.assertIn("Resumed session", text)
        self.assertIn("Session not found: missing", text)
        self.assertIn("Deleted session", text)
        self.assertNotEqual(other.id, s.id)  # deleting the live session starts a new one
        self.assertEqual(self.store.list(), [])

    def test_parse_args_sessions(self):
        a = nada.parse_args(["--session", "abc123", "do it"])
        self.assertEqual(a.session, "abc123")
        self.assertEqual(a.task, ["do it"])
        a = nada.parse_args(["--continue"])
        self.assertTrue(a.continue_session)
        a = nada.parse_args(["--session-dir", "/tmp/s", "--no-autosave"])
        self.assertEqual(a.session_dir, "/tmp/s")
        self.assertTrue(a.no_autosave)
        with self.assertRaises(SystemExit):  # mutually exclusive
            nada.parse_args(["--session", "abc", "--continue"])

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

        a = nada.parse_args(["--model", "openai/gpt-5.2", "do", "it"])
        self.assertEqual(a.model, "openai/gpt-5.2")
        self.assertEqual(a.task, ["do", "it"])
        self.assertIsNone(nada.parse_args([]).model)


class TestTerminalUI(unittest.TestCase):
    """Codex-style TUI: boxed banner, spinner, collapsed ✓/✗ tool lines,
    diff approvals, token footer, /help /status /verbose, tty-only colour."""

    class _Tty(io.StringIO):
        def isatty(self):
            return True

    def test_use_color_only_on_tty_and_never_with_no_color(self):
        self.assertTrue(nada._use_color(self._Tty()))
        self.assertFalse(nada._use_color(io.StringIO()))
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            self.assertFalse(nada._use_color(self._Tty()))
        with patch.dict(os.environ, {"TERM": "dumb"}):
            self.assertFalse(nada._use_color(self._Tty()))

    def test_style_and_strip_ansi(self):
        with patch.object(nada, "_use_color", lambda stream=None: True):
            self.assertEqual(nada._style("x", "1"), "\x1b[1mx\x1b[0m")
        with patch.object(nada, "_use_color", lambda stream=None: False):
            self.assertEqual(nada._style("x", "1"), "x")
        self.assertEqual(nada._strip_ansi("\x1b[32mhi\x1b[0m"), "hi")

    def test_format_duration(self):
        self.assertEqual(nada.format_duration(0), "0ms")
        self.assertEqual(nada.format_duration(0.04), "40ms")
        self.assertEqual(nada.format_duration(1.0), "1s")
        self.assertEqual(nada.format_duration(1.5), "1.5s")
        self.assertEqual(nada.format_duration(65), "1m 5s")

    def test_tool_summary(self):
        self.assertEqual(nada.tool_summary("bash", {"command": "ls -R"}), "Ran ls -R")
        self.assertEqual(nada.tool_summary("bash", {"command": "ls -R"}, running=True),
                         "Running ls -R")
        self.assertEqual(nada.tool_summary("read_file", {"path": "nada.py"}), "Read nada.py")
        self.assertEqual(nada.tool_summary("write_file", {"path": "x.py"}), "Wrote x.py")
        self.assertEqual(nada.tool_summary("edit_file", {"path": "x.py"}), "Edited x.py")
        long_summary = nada.tool_summary("bash", {"command": "x " * 100})
        self.assertTrue(long_summary.endswith("…"))
        self.assertLessEqual(len(long_summary), 80)

    def test_format_tool_result_collapses_bash_output(self):
        result = "$ ls\n[exit 0]\n" + "\n".join(f"line{i}" for i in range(20))
        out = nada.format_tool_result("bash", {"command": "ls"}, result, ok=True)
        self.assertTrue(out.startswith("✓ Ran ls"))
        self.assertIn("line0", out)
        self.assertIn("line11", out)       # TOOL_OUTPUT_MAX_LINES shown
        self.assertNotIn("line12", out)    # ... the rest collapsed
        self.assertIn("8 more lines", out)
        self.assertNotIn("$ ls", out)      # the `$ cmd` header is not repeated
        verbose = nada.format_tool_result("bash", {"command": "ls"}, result,
                                          ok=True, verbose=True)
        self.assertIn("line19", verbose)
        self.assertNotIn("more lines", verbose)

    def test_format_tool_result_failed_and_read(self):
        out = nada.format_tool_result("bash", {"command": "ls"},
                                      "$ ls\n[exit 3]\nboom", ok=False)
        self.assertIn("✗ Ran ls · exit 3", out)
        self.assertIn("boom", out)
        out = nada.format_tool_result("read_file", {"path": "x.py"},
                                      "    1 | a\n    2 | b\n    3 | c", ok=True)
        self.assertIn("✓ Read x.py · 3 lines", out)
        self.assertNotIn("| a", out)  # file contents are not dumped
        out = nada.format_tool_result("bash", {"command": "ls"}, "Error: boom", ok=False)
        self.assertIn("✗ Ran ls", out)
        self.assertIn("Error: boom", out)

    def test_format_diff(self):
        out = nada.format_diff("a\nb\nc", "a\nB\nc", "x.py")
        self.assertIn("--- x.py", out)
        self.assertIn("+++ x.py", out)
        self.assertIn("-b", out)
        self.assertIn("+B", out)
        self.assertEqual(nada.format_diff("a", "a", "x.py"), "(no changes)")
        big = nada.format_diff("\n".join(f"old{i}" for i in range(30)),
                               "\n".join(f"new{i}" for i in range(30)),
                               "x.py", max_lines=5)
        self.assertIn("more diff lines", big)

    def test_preview_is_command_or_diff(self):
        self.assertEqual(nada._preview("bash", {"command": "rm -rf /"}), "$ rm -rf /")
        out = nada._preview("edit_file", {"path": "x.py", "old_string": "old",
                                          "new_string": "new"})
        self.assertIn("-old", out)
        self.assertIn("+new", out)
        out = nada._preview("write_file", {"path": "brand_new_file_xyz.py",
                                           "content": "hello\n"})
        self.assertIn("+hello", out)

    def test_repl_prompt_bare_by_default_mode_only_as_fallback(self):
        p = perms("plan")
        # The mode lives in the statusline on a tty, so the prompt is bare.
        self.assertEqual(nada._repl_prompt(p, stream=io.StringIO()), "> ")
        with patch.object(nada, "_use_color", lambda stream=None: True):
            coloured = nada._repl_prompt(p, stream=io.StringIO())
        self.assertIn("\x1b[", coloured)
        self.assertEqual(nada._strip_ansi(coloured), "> ")
        # Only when no statusline will be drawn does the prompt carry it.
        self.assertEqual(nada._repl_prompt(p, stream=io.StringIO(),
                                           show_mode=True), "[plan] > ")
        with patch.object(nada, "_use_color", lambda stream=None: True):
            coloured = nada._repl_prompt(p, stream=io.StringIO(), show_mode=True)
        self.assertIn("\x1b[", coloured)
        self.assertEqual(nada._strip_ansi(coloured), "[plan] > ")

    def test_format_banner_is_a_box_of_the_given_width(self):
        banner = nada.format_banner(model="meta/muse-ember", root="/r", mode="auto",
                                    effort="high", session_id="s1",
                                    session_path="/s/s1.json",
                                    agents=["AGENTS.md"], width=60)
        lines = banner.splitlines()
        self.assertTrue(all(len(nada._strip_ansi(l)) == 60 for l in lines))
        self.assertTrue(lines[0].startswith("╭─") and lines[0].endswith("╮"))
        self.assertTrue(lines[-1].startswith("╰─") and lines[-1].endswith("╯"))
        for needle in ["nada", "/r", "AGENTS.md: AGENTS.md", "/help for commands"]:
            self.assertIn(needle, banner)
        # Live state is the statusline's job, not the banner's: none of it
        # (passed above for compatibility) may be repeated here.
        for duplicated in ["meta/muse-ember", "auto", "high", "s1"]:
            self.assertNotIn(duplicated, banner)

    def test_format_statusline(self):
        session = type("S", (), {"id": "s1", "turns": 3})()
        bar = nada.format_statusline(model="meta/muse-ember", root="/home/u/nada",
                                     mode="auto", effort="high", session=session,
                                     usage={"prompt_tokens": 1234,
                                            "completion_tokens": 567},
                                     width=200)
        self.assertEqual(bar, "meta/muse-ember · nada · auto · effort: high · "
                              "session: s1 · 3 turns · tokens: 1,234 in · 567 out")
        self.assertNotIn("\x1b[", bar)  # plain off-tty
        singular = nada.format_statusline(model="m", root="/r", mode="plan",
                                          effort=None, session_id="s",
                                          turns=1, usage={}, width=200)
        self.assertTrue(singular.endswith("· 1 turn"))
        self.assertIn("effort: (model default)", singular)
        self.assertNotIn("tokens:", singular)  # no usage, no segment

    def test_format_statusline_drops_low_priority_segments_when_narrow(self):
        bar = nada.format_statusline(model="meta/muse-ember", root="/home/u/nada",
                                     mode="auto", effort="high", session_id="s1",
                                     turns=3, usage={"prompt_tokens": 1234,
                                                     "completion_tokens": 567},
                                     width=40)
        self.assertLessEqual(len(nada._strip_ansi(bar)), 40)
        self.assertIn("meta/muse-ember", bar)   # core segments survive
        self.assertIn("auto", bar)
        self.assertNotIn("tokens:", bar)        # usage dropped first
        self.assertNotIn("session:", bar)       # ... then session

    def test_format_statusline_coloured_mode_and_globals(self):
        with patch.object(nada, "_use_color", lambda stream=None: True), \
             patch.object(nada, "MODEL", "m2"), patch.object(nada, "EFFORT", "low"), \
             patch.object(nada, "_LAST_USAGE", {"prompt_tokens": 1,
                                                "completion_tokens": 2}):
            bar = nada.format_statusline(root="/x/proj", mode="yolo", width=200)
        self.assertIn("\x1b[", bar)
        self.assertEqual(nada._strip_ansi(bar),
                         "m2 · proj · yolo · effort: low · tokens: 1 in · 2 out")
        # ANSI codes must not count towards the width.
        with patch.object(nada, "_use_color", lambda stream=None: True):
            narrow = nada.format_statusline(model="meta/muse-ember", root="/x/nada",
                                            mode="auto", effort="high", width=60)
        self.assertLessEqual(len(nada._strip_ansi(narrow)), 60)
        self.assertIn("effort: high", nada._strip_ansi(narrow))

    def test_resolve_statusline_does_not_retruncate_a_formatted_bar(self):
        with patch.object(nada, "_use_color", lambda stream=None: True):
            bar = nada.format_statusline(model="meta/muse-ember", root="/x/nada",
                                         mode="auto", effort="high",
                                         session_id="s1", turns=2, width=200)
        resolved = nada._resolve_statusline(lambda: bar, perms("auto"))
        self.assertEqual(resolved, bar)  # colour + length preserved
        self.assertEqual(nada._resolve_statusline("a\nb", perms()), "a b")

    def test_read_line_raw_draws_statusline_under_prompt_on_tty(self):
        p = perms("default")
        out = self._Tty()
        with patch.dict(os.environ, {"TERM": "xterm"}):
            line = nada._read_line_raw(p, io.BytesIO(b"hi\r"), out,
                                       statusline="STATUSBAR")
        self.assertEqual(line, "hi")
        self.assertIn("STATUSBAR", out.getvalue())
        self.assertNotIn("[default]", out.getvalue())  # mode only in the bar
        self.assertIn("\x1b[1A", out.getvalue())     # cursor back up to prompt
        self.assertIn("\r\n\x1b[K", out.getvalue())  # submit clears the bar line

    def test_read_line_raw_no_statusline_off_tty(self):
        p = perms("default")
        out = io.StringIO()
        line = nada._read_line_raw(p, io.BytesIO(b"hi\r"), out,
                                   statusline="STATUSBAR")
        self.assertEqual(line, "hi")
        self.assertNotIn("STATUSBAR", out.getvalue())
        self.assertNotIn("\x1b[1A", out.getvalue())
        # No bar to carry the mode, so the prompt falls back to `[mode] > `.
        self.assertIn("[default] > ", out.getvalue())

    def test_read_repl_line_fallback_prompt_carries_mode_only_without_bar(self):
        p = perms("plan")
        prompts, printed = [], []
        with patch.object(nada, "_is_tty", lambda: False), \
             patch.object(nada, "termios", None), \
             patch.object(nada, "_statusline_enabled", lambda stream=None: False), \
             patch("builtins.input", lambda prompt="": prompts.append(prompt) or "task"):
            self.assertEqual(nada.read_repl_line(p, statusline="STATUSBAR"), "task")
        self.assertEqual(prompts, ["[plan] > "])  # no bar: mode in the prompt
        self.assertEqual(printed, [])
        prompts.clear()
        with patch.object(nada, "_is_tty", lambda: False), \
             patch.object(nada, "termios", None), \
             patch.object(nada, "_statusline_enabled", lambda stream=None: True), \
             patch("builtins.print", lambda *a, **k: printed.append(a[0])), \
             patch("builtins.input", lambda prompt="": prompts.append(prompt) or "task"):
            self.assertEqual(nada.read_repl_line(p, statusline="STATUSBAR"), "task")
        self.assertEqual(prompts, ["> "])          # bar above: bare prompt
        self.assertEqual(printed, ["STATUSBAR"])

    def test_read_line_raw_statusline_refreshes_on_mode_cycle(self):
        p = perms("default")
        out = self._Tty()
        with tempfile.TemporaryDirectory() as tmp:  # keep cycle-persistence off the real file
            env = {"TERM": "xterm", "NADA_SETTINGS_FILE": os.path.join(tmp, "settings.json")}
            with patch.dict(os.environ, env):
                line = nada._read_line_raw(
                    p, io.BytesIO(b"\x1b[27;5;9~x\r"), out,
                    statusline=lambda: f"mode={p.mode}")
        self.assertEqual((line, p.mode), ("x", "accept-edits"))
        self.assertIn("mode=accept-edits", out.getvalue())

    def test_format_usage(self):
        self.assertEqual(nada.format_usage({"prompt_tokens": 1234,
                                            "completion_tokens": 567}),
                         "tokens: 1,234 in · 567 out")
        self.assertIsNone(nada.format_usage({}))
        self.assertIsNone(nada.format_usage(None))

    def test_spinner_is_a_noop_off_tty(self):
        out = io.StringIO()
        with nada.Spinner("Working", stream=out):
            pass
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(nada.spinner_frame(0), nada.SPINNER_FRAMES[0])

    def test_help_status_verbose_commands(self):
        out = io.StringIO()
        with patch("sys.stdout", out):
            nada.handle_help_command(["/help"])
        for needle in ["/status", "/verbose", "/mode", "/effort", "Ctrl+Tab"]:
            self.assertIn(needle, out.getvalue())

        out = io.StringIO()
        reasoning = nada.parse_model_reasoning(
            {"reasoning": {"supported_efforts": ["high"], "default_effort": "high"}})
        with patch("sys.stdout", out):
            nada.handle_status_command(
                session=type("S", (), {"id": "s1", "turns": 3,
                                       "path": lambda self=None: "/s/s1.json"})(),
                reasoning=reasoning)
        for needle in ["Model:", "Permissions:", "high", "Session: s1", "Turns: 3"]:
            self.assertIn(needle, out.getvalue())

        out = io.StringIO()
        with patch.object(nada, "VERBOSE", False), patch("sys.stdout", out):
            self.assertFalse(nada.handle_verbose_command(["/verbose"]))
            self.assertTrue(nada.handle_verbose_command(["/verbose", "on"]))
            self.assertTrue(nada.VERBOSE)
            self.assertFalse(nada.handle_verbose_command(["/verbose", "off"]))
        self.assertIn("Verbose output: off", out.getvalue())

    def test_ask_accepts_numbered_answers(self):
        self.assertTrue(perms("ask", ["1"]).check("bash", {"command": "make build"}))
        p = perms("ask", ["2"])
        self.assertTrue(p.check("bash", {"command": "make build"}))
        self.assertIn(("bash", "make build"), p.allowlist)
        self.assertFalse(perms("ask", ["3"]).check("bash", {"command": "make build"}))

    def test_parse_args_no_color_and_verbose(self):
        a = nada.parse_args(["--no-color", "--verbose", "do it"])
        self.assertTrue(a.no_color)
        self.assertTrue(a.verbose)
        self.assertEqual(a.task, ["do it"])
        a = nada.parse_args(["do it"])
        self.assertFalse(a.no_color)
        self.assertFalse(a.verbose)

    def _run_turn(self, handler_result, *, p=None, usage=None, verbose=False):
        responses = iter([
            {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "1", "function": {"name": "bash",
                                         "arguments": '{"command": "make build"}'}}]}}],
             **({"usage": usage} if usage else {})},
            {"choices": [{"message": {"content": "done"}}]},
        ])
        out = io.StringIO()
        with patch.object(nada, "call_model", lambda messages: next(responses)), \
             patch.object(nada, "PERMS", p or perms("yolo")), \
             patch.object(nada, "HANDLERS", {"bash": lambda a: handler_result}), \
             patch.object(nada, "VERBOSE", verbose), \
             patch("sys.stdout", out):
            messages = [{"role": "system", "content": ""}]
            nada.run_turns(messages)
        return out.getvalue(), messages

    def test_run_turns_collapsed_output_and_token_footer(self):
        result = "$ ls\n[exit 0]\n" + "\n".join(f"line{i}" for i in range(20))
        out, _ = self._run_turn(result, usage={"prompt_tokens": 10,
                                               "completion_tokens": 5})
        self.assertIn("✓ Ran make build", out)
        self.assertIn("8 more lines", out)
        self.assertIn("tokens: 10 in · 5 out", out)
        self.assertEqual(nada._LAST_USAGE,
                         {"prompt_tokens": 10, "completion_tokens": 5})
        self.assertIn("done", out)
        self.assertNotIn('{"command"', out)  # no raw argument dump

    def test_run_turns_denied_and_failed_tools(self):
        out, _ = self._run_turn("unused", p=perms("ask", ["n"]))
        self.assertIn("✗ Ran make build", out)
        self.assertIn("PermissionDenied", out)
        out, _ = self._run_turn("$ ls\n[exit 2]\nboom")
        self.assertIn("✗ Ran make build · exit 2", out)
        self.assertIn("boom", out)

    def test_run_turns_verbose_shows_full_output(self):
        result = "$ ls\n[exit 0]\n" + "\n".join(f"line{i}" for i in range(20))
        out, _ = self._run_turn(result, verbose=True)
        self.assertIn("line19", out)
        self.assertNotIn("more lines", out)


class TestAgentsMd(unittest.TestCase):
    """AGENTS.md per https://agents.md: root + nested, exact filename,
    plain Markdown, closest to the edited file wins, user chat overrides."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        from pathlib import Path
        self.root = Path(self._tmp.name)

    def _write(self, rel, text):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def test_root_and_nested_ordering(self):
        self._write("AGENTS.md", "# root\n")
        self._write("pkg/AGENTS.md", "# pkg\n")
        self._write("pkg/sub/AGENTS.md", "# sub\n")
        self._write("other/AGENTS.md", "# other\n")
        found = [p.relative_to(self.root).as_posix()
                 for p in nada.find_agents_files(self.root)]
        # Root first, then shallow-to-deep (parents before descendants).
        self.assertEqual(found, ["AGENTS.md", "other/AGENTS.md",
                                 "pkg/AGENTS.md", "pkg/sub/AGENTS.md"])
        loaded = nada.load_agents_instructions(self.root)
        self.assertEqual([rel for rel, _ in loaded], found)
        self.assertEqual(loaded[0][1], "# root\n")

    def test_only_exact_agents_md_filename(self):
        self._write("AGENTS.md", "# root\n")
        for name in ["AGENT.md", "agents.md", "CLAUDE.md", "AGENTS.md.bak", "xAGENTS.md"]:
            self._write(name, "# not the spec filename\n")
            self._write(f"sub/{name}", "# not the spec filename\n")
        (self.root / "dir/AGENTS.md").mkdir(parents=True)  # a directory, not a file
        found = [p.relative_to(self.root).as_posix()
                 for p in nada.find_agents_files(self.root)]
        self.assertEqual(found, ["AGENTS.md"])

    def test_no_agents_md_leaves_prompt_unchanged(self):
        self.assertEqual(nada.find_agents_files(self.root), [])
        self.assertEqual(nada.load_agents_instructions(self.root), [])
        prompt = nada.build_system_prompt(self.root)
        self.assertNotIn("AGENTS.md files in this repository follow", prompt)
        self.assertIn("You are a coding agent", prompt)

    def test_build_includes_verbatim_labelled_and_precedence(self):
        self._write("AGENTS.md", "# root\nrun `make root`\n")
        self._write("pkg/AGENTS.md", "# pkg\nrun `make pkg`\n")
        prompt = nada.build_system_prompt(self.root)
        self.assertIn("--- AGENTS.md: AGENTS.md ---\n# root\nrun `make root`", prompt)
        self.assertIn("--- AGENTS.md: pkg/AGENTS.md ---\n# pkg\nrun `make pkg`", prompt)
        # Nested (closest) appears after its parent.
        self.assertLess(prompt.index("run `make root`"), prompt.index("run `make pkg`"))
        # Spec precedence rule, verbatim in substance.
        self.assertIn("closest AGENTS.md to the file being edited wins", prompt)
        self.assertIn("explicit user instructions in the chat override AGENTS.md", prompt)

    def test_session_system_prompt_uses_agents_and_regenerates(self):
        self._write("AGENTS.md", "# root instructions\n")
        self._write("pkg/AGENTS.md", "# nested instructions\n")
        store = nada.SessionStore(str(self.root / "sessions"))
        with patch.object(nada, "ROOT", self.root):
            session = nada.Session(store)
            self.assertIn("# root instructions", session.messages[0]["content"])
            self.assertIn("# nested instructions", session.messages[0]["content"])
            # Living documentation: a later edit is picked up on reset.
            self._write("AGENTS.md", "# updated root\n")
            session.reset()
            self.assertIn("# updated root", session.messages[0]["content"])
            self.assertNotIn("# root instructions", session.messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
