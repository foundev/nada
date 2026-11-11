#!/usr/bin/env python3
import unittest
from unittest.mock import patch

import nada


def perms(mode="ask", answers=None, tty=True):
    answers = iter(answers or [])
    def fake_input(prompt=""):
        try:
            return next(answers)
        except StopIteration:
            raise AssertionError("input() should not have been called")
    return nada.Permissions(mode=mode, input_fn=fake_input, isatty_fn=lambda: tty)


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
        for mode in ["ask", "auto", "deny"]:
            self.assertTrue(perms(mode).check("read_file", {"path": "nada.py"}))

    def test_read_only_bash_always_allowed(self):
        for mode in ["ask", "auto", "deny"]:
            self.assertTrue(perms(mode).check("bash", {"command": "ls -R"}))

    def test_auto_allows_everything(self):
        p = perms("auto")
        self.assertTrue(p.check("write_file", {"path": "x", "content": "y"}))
        self.assertTrue(p.check("bash", {"command": "rm -rf build"}))

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


class TestParseArgs(unittest.TestCase):
    def test_task_and_flags(self):
        a = nada.parse_args(["--yes", "do", "the", "thing"])
        self.assertTrue(a.yes)
        self.assertEqual(a.task, ["do", "the", "thing"])
        self.assertIsNone(a.permission_mode)

        a = nada.parse_args(["--permission-mode", "deny"])
        self.assertEqual(a.permission_mode, "deny")
        self.assertEqual(a.task, [])

        a = nada.parse_args(["hello"])
        self.assertFalse(a.yes)
        self.assertEqual(a.task, ["hello"])


if __name__ == "__main__":
    unittest.main()
