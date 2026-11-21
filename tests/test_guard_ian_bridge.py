"""
Tests for the GUARD.IAN bridge. No network, no money: the A2A endpoint, the Metaculus client and the
research/summary/parser models are all stubbed.

Run: uv run python -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests
from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    ForecastReport,
    GeneralLlm,
    MetaculusClient,
    MultipleChoiceQuestion,
    NumericQuestion,
)

import guard_ian_bridge as bridge
import main
from guard_ian_bridge import (
    GuardIanBadReply,
    GuardIanDeclined,
    GuardIanError,
    GuardIanLlm,
    GuardIanTransportError,
    SaveOnlyMetaculusClient,
)

REPO = Path(__file__).resolve().parent.parent
GOOD_BINARY_REPLY = (
    "(a) About four weeks until the release.\n"
    "(b) Status quo: recent prints were near 3%, per https://www.bls.gov/cpi/ .\n"
    "My take: the base rate for a move of this size in one month is low.\n"
    "Probability: 37%"
)


def completed_task(text: str) -> dict:
    return {
        "id": "task-1",
        "status": {"state": "completed", "timestamp": "2026-11-20T00:00:00Z"},
        "artifacts": [{"parts": [{"type": "text", "text": text}], "index": 0}],
    }


class FakeTransport:
    """Stands in for the A2A endpoint. Records every request it gets."""

    def __init__(self, reply: str | None = GOOD_BINARY_REPLY, error: Exception | None = None) -> None:
        self.reply, self.error, self.requests = reply, error, []

    async def __call__(self, text: str) -> dict:
        self.requests.append(text)
        if self.error:
            raise self.error
        return completed_task(self.reply or "")


class FakeLlm(GeneralLlm):
    """Research, summary and parser stand-in. The parser copies the 'Probability: NN%' line into JSON."""

    def __init__(self, role: str) -> None:
        super().__init__(model=f"fake/{role}", temperature=None)
        self.role, self.calls = role, 0

    async def invoke(self, prompt, system_prompt=None) -> str:
        self.calls += 1
        if self.role == "parser":
            match = re.search(r"Probability:\s*(\d+(?:\.\d+)?)\s*%", str(prompt))
            return json.dumps({"prediction_in_decimal": float(match.group(1)) / 100})
        return f"{self.role} text. Source: https://www.bls.gov/cpi/"


def support_llms() -> dict:
    return {role: FakeLlm(role) for role in ("researcher", "summarizer", "parser")}


class NetworkGuard:
    """Fails the test if anything tries an HTTP request through requests."""

    def __enter__(self):
        def refuse(*args, **kwargs):
            raise AssertionError(f"unexpected HTTP request: {args[:2]}")

        self.patches = [mock.patch.object(requests.Session, "request", side_effect=refuse)]
        for verb in ("get", "post", "put", "patch", "delete", "request"):
            self.patches.append(mock.patch.object(requests, verb, side_effect=refuse))
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


class SaveOnlyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"GUARD_IAN_SAVE_DIR": self.tmp.name})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def test_save_only_runs_the_whole_path_and_makes_zero_metaculus_writes(self) -> None:
        transport, llms = FakeTransport(), support_llms()
        bot = main.build_guard_ian_bot(save_only=True, transport=transport, support_llms=llms)
        self.assertFalse(bot.publish_reports_to_metaculus)
        self.assertIsInstance(bot.metaculus_client, SaveOnlyMetaculusClient)
        self.assertEqual(bot.predictions_per_research_report, 1)

        write_spies = {
            name: mock.patch.object(SaveOnlyMetaculusClient, name, autospec=True)
            for name in SaveOnlyMetaculusClient.WRITE_METHODS
        }
        started = {name: p.start() for name, p in write_spies.items()}
        try:
            with NetworkGuard():
                reports = main.run_save_only(bot)
        finally:
            for p in write_spies.values():
                p.stop()

        for name, spy in started.items():
            spy.assert_not_called()
        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertIsInstance(report, ForecastReport, msg=repr(report))
        self.assertAlmostEqual(report.prediction, 0.37)
        self.assertIn("Probability: 37%", report.explanation)
        self.assertIn("guard-ian/a2a", report.explanation)  # the public comment names the forecaster
        self.assertEqual(len(transport.requests), 1)  # one A2A call
        self.assertTrue(transport.requests[0].startswith(bridge.REQUEST_HEADER))
        self.assertIn("question_type: binary", transport.requests[0])

        saved_reports = glob.glob(os.path.join(self.tmp.name, "Forecasts-for-*.json"))
        self.assertEqual(len(saved_reports), 1)
        lines = Path(self.tmp.name, "guard-ian-a2a-replies.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(lines[0])["outcome"], "ok")
        self.assertEqual(json.loads(lines[0])["a2a_task"]["status"]["state"], "completed")

    def test_save_only_with_a_fully_stubbed_client_records_no_calls_at_all(self) -> None:
        client = mock.MagicMock(spec=SaveOnlyMetaculusClient)
        client.__class__ = SaveOnlyMetaculusClient
        bot = main.build_guard_ian_bot(
            save_only=True, transport=FakeTransport(), support_llms=support_llms(), metaculus_client=client
        )
        with NetworkGuard():
            reports = main.run_save_only(bot)
        self.assertIsInstance(reports[0], ForecastReport)
        self.assertEqual(client.method_calls, [])
        self.assertEqual(client.mock_calls, [])

    def test_failure_skips_the_question_with_no_fallback_and_no_retry(self) -> None:
        for error in (
            GuardIanTransportError("A2A helper exited 3: Request timed out"),
            None,  # bad format: see the reply below
        ):
            transport = FakeTransport(reply="I think it is likely.", error=error)
            llms = support_llms()
            client = mock.MagicMock(spec=SaveOnlyMetaculusClient)
            client.__class__ = SaveOnlyMetaculusClient
            bot = main.build_guard_ian_bot(
                save_only=True, transport=transport, support_llms=llms, metaculus_client=client
            )
            with NetworkGuard():
                reports = main.run_save_only(bot)
            self.assertIsInstance(reports[0], BaseException)
            self.assertEqual(len(transport.requests), 1)  # asked once, never re-asked
            self.assertEqual(llms["parser"].calls, 0)  # nothing parsed, nothing submitted
            self.assertEqual(client.mock_calls, [])

    def test_decline_skips_the_question(self) -> None:
        bot = main.build_guard_ian_bot(
            save_only=True, transport=FakeTransport(reply="DECLINED: price question"), support_llms=support_llms()
        )
        with NetworkGuard():
            reports = main.run_save_only(bot)
        self.assertIsInstance(reports[0], BaseException)
        self.assertIn("DECLINED", str(reports[0]))

    def test_run_save_only_refuses_a_publishing_bot(self) -> None:
        bot = main.build_guard_ian_bot(
            save_only=False, transport=FakeTransport(), support_llms=support_llms(),
            metaculus_client=mock.MagicMock(spec=MetaculusClient),
        )
        with self.assertRaises(SystemExit):
            main.run_save_only(bot)


class OneBotTests(unittest.TestCase):
    def test_live_bot_forecasts_only_with_guard_ian(self) -> None:
        bot = main.build_guard_ian_bot(
            save_only=False, transport=FakeTransport(), support_llms=support_llms(),
            metaculus_client=mock.MagicMock(spec=MetaculusClient),
        )
        self.assertIsInstance(bot.get_llm("default"), GuardIanLlm)
        self.assertTrue(bot.publish_reports_to_metaculus)
        self.assertEqual(bot.predictions_per_research_report, 5)
        self.assertTrue(bot.skip_previously_forecasted_questions)

    def test_the_forecaster_cannot_be_replaced(self) -> None:
        with self.assertRaises(ValueError):
            main.build_guard_ian_bot(save_only=True, support_llms={"default": FakeLlm("stock")})

    def test_a_bot_with_a_stock_forecaster_refuses_to_run(self) -> None:
        bot = main.build_guard_ian_bot(save_only=True, transport=FakeTransport(), support_llms=support_llms())
        bot.set_llm(GeneralLlm(model="openai/gpt-6-luna"), "default")
        with self.assertRaises(SystemExit):
            bridge.assert_guard_ian_is_the_only_forecaster(bot)

    def test_guard_ian_llm_never_calls_a_model_without_a_question(self) -> None:
        llm = GuardIanLlm(transport=FakeTransport())
        with mock.patch.object(GeneralLlm, "_mockable_direct_call_to_model") as stock:
            with self.assertRaises(GuardIanError):
                asyncio.run(llm.invoke("hello"))
            stock.assert_not_called()

    def test_guard_ian_llm_does_not_fall_back_to_a_stock_model(self) -> None:
        llm = GuardIanLlm(transport=FakeTransport(error=GuardIanTransportError("down")))
        token = bridge.CURRENT_QUESTION.set(bridge.synthetic_question())
        try:
            with mock.patch.object(GeneralLlm, "_mockable_direct_call_to_model") as stock:
                with self.assertRaises(GuardIanTransportError):
                    asyncio.run(llm.invoke("forecast this"))
                stock.assert_not_called()
        finally:
            bridge.CURRENT_QUESTION.reset(token)

    def test_no_framework_script_is_disabled(self) -> None:
        result = subprocess.run(
            [sys.executable, "main_with_no_framework.py"], cwd=REPO, capture_output=True, text=True,
            env={**os.environ, "METACULUS_TOKEN": "", "OPENAI_API_KEY": ""}, timeout=300,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("disabled in this fork", result.stderr)

    def test_main_without_live_defaults_to_save_only(self) -> None:
        result = subprocess.run(
            [sys.executable, "main.py", "--help"], cwd=REPO, capture_output=True, text=True, timeout=300
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--live", result.stdout)
        self.assertIn("SAVE-ONLY", result.stdout)


class SaveOnlyClientTests(unittest.TestCase):
    def test_every_write_method_refuses(self) -> None:
        client = SaveOnlyMetaculusClient(token="not-a-real-token")
        with NetworkGuard():
            for name in SaveOnlyMetaculusClient.WRITE_METHODS:
                with self.assertRaises(RuntimeError, msg=name):
                    getattr(client, name)(1, 2)

    def test_write_list_covers_every_method_of_the_library_that_sends_a_write_request(self) -> None:
        import inspect

        reads_sent_as_post = {"_get_acting_user_forecasts"}  # bulk-forecast-read: a read that uses POST
        senders = {
            name
            for name, fn in inspect.getmembers(MetaculusClient, inspect.isfunction)
            if re.search(r"requests\.(post|put|patch|delete)\(", inspect.getsource(fn))
        }
        self.assertIn("post_question_comment", senders)  # the scan itself works
        missing = senders - reads_sent_as_post - set(SaveOnlyMetaculusClient.WRITE_METHODS)
        self.assertEqual(missing, set(), "a new Metaculus write method must be added to WRITE_METHODS")
        # The prediction posters send through _post_question_prediction; they are refused too.
        for name in ("post_binary_question_prediction", "post_numeric_question_prediction",
                     "post_multiple_choice_question_prediction"):
            self.assertIn(name, SaveOnlyMetaculusClient.WRITE_METHODS)


def mc_question() -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(question_text="Which?", options=["Red", "Green", "Blue team"])


def numeric_question() -> NumericQuestion:
    return NumericQuestion(
        question_text="How many?", upper_bound=500, lower_bound=0, open_upper_bound=True, open_lower_bound=False
    )


def date_question() -> DateQuestion:
    from datetime import datetime, timezone

    return DateQuestion(
        question_text="When?",
        upper_bound=datetime(2030, 1, 1, tzinfo=timezone.utc),
        lower_bound=datetime(2026, 1, 1, tzinfo=timezone.utc),
        open_upper_bound=True,
        open_lower_bound=False,
    )


class ValidationTests(unittest.TestCase):
    binary = BinaryQuestion(question_text="Will it?")

    def test_binary(self) -> None:
        bridge.validate_reply(GOOD_BINARY_REPLY, self.binary)
        bridge.validate_reply("Reasoning.\n**Probability: 12.5%**", self.binary)
        for bad in (
            "Reasoning.\nProbability: 37%\nStandGuard.",  # something after the answer
            "Reasoning.\nProbability: 150%",
            "Reasoning.\nProbability: 0.37",
            "Reasoning only.",
        ):
            with self.assertRaises(GuardIanBadReply, msg=bad):
                bridge.validate_reply(bad, self.binary)

    def test_decline(self) -> None:
        with self.assertRaises(GuardIanDeclined):
            bridge.validate_reply("DECLINED: conflict of interest", self.binary)

    def test_private_markers(self) -> None:
        for leak in (
            "Signed SIG_K1_abc.\nProbability: 40%",
            "The key PVT_K1_abc.\nProbability: 40%",
            "From the team notebook.\nProbability: 40%",
            "GUARD never says safe.\nProbability: 40%",
            "Per guard-ian-forecaster rules.\nProbability: 40%",
        ):
            with self.assertRaises(GuardIanBadReply, msg=leak):
                bridge.validate_reply(leak, self.binary)
        with mock.patch.dict(os.environ, {"GUARD_IAN_PRIVATE_TERMS": "examplewallet1, Example Person"}):
            with self.assertRaises(GuardIanBadReply):
                bridge.validate_reply("As example person said.\nProbability: 40%", self.binary)
            bridge.validate_reply("Nothing private.\nProbability: 40%", self.binary)

    def test_multiple_choice(self) -> None:
        q = mc_question()
        bridge.validate_reply("Why.\nRed: 50%\nGreen: 30%\nBlue team: 20%", q)
        for bad in (
            "Why.\nGreen: 30%\nRed: 50%\nBlue team: 20%",  # wrong order
            "Why.\nRed: 50%\nGreen: 30%\nBlue team: 30%",  # sums to 110
            "Why.\nRed: 50%\nGreen: 50%",  # missing option
        ):
            with self.assertRaises(GuardIanBadReply, msg=bad):
                bridge.validate_reply(bad, q)

    def test_numeric(self) -> None:
        q = numeric_question()
        good = "Why.\nPercentile 10: 10 (lowest number value)\nPercentile 20: 20\nPercentile 40: 40\n" \
               "Percentile 60: 60\nPercentile 80: 1,200\nPercentile 90: 1500 (highest number value)"
        bridge.validate_reply(good, q)
        for bad in (
            good.replace("Percentile 60: 60", "Percentile 60: 30"),  # not increasing
            good.replace("Percentile 90: 1500", "Percentile 90: 1.5e3"),  # scientific notation
            good.replace("Percentile 20", "Percentile 25"),
        ):
            with self.assertRaises(GuardIanBadReply, msg=bad):
                bridge.validate_reply(bad, q)

    def test_date(self) -> None:
        q = date_question()
        dates = ("2027-01-01", "2027-03-01", "2027-06-01", "2027-09-01", "2028-01-01", "2028-06-01")
        good = "Why.\n" + "\n".join(f"Percentile {p}: {d}" for p, d in zip(bridge.PERCENTILES, dates))
        bridge.validate_reply(good, q)
        bridge.validate_reply(good.replace("2027-01-01", "2027-01-01 (oldest date)"), q)
        for bad in (good.replace("2027-06-01", "2026-06-01"), good.replace("2027-03-01", "March 2027")):
            with self.assertRaises(GuardIanBadReply, msg=bad):
                bridge.validate_reply(bad, q)

    def test_task_must_be_completed_with_text(self) -> None:
        self.assertEqual(bridge.reply_text_from_task(completed_task("hi")), "hi")
        for task in ({"status": {"state": "failed"}}, completed_task("   "), {}):
            with self.assertRaises(GuardIanBadReply):
                bridge.reply_text_from_task(task)

    def test_request_shape_and_size_limit(self) -> None:
        request = bridge.build_request("PROMPT", mc_question())
        self.assertTrue(request.startswith("GUARD.IAN FORECAST REQUEST v1\nquestion_type: multiple_choice\n"))
        self.assertIn("PROMPT", request)
        with self.assertRaises(GuardIanBadReply):
            bridge.build_request("x" * (bridge.MAX_REQUEST_CHARS + 1), self.binary)


if __name__ == "__main__":
    unittest.main()
