"""
GUARD.IAN bridge: makes GUARD.IAN, an AI agent on XPR Network, the ONLY forecaster of this bot.

How a forecast is made
- The template does research (OpenAI web search), as before.
- Every forecast the template asks for goes to `GuardIanLlm`, the bot's "default" model. It sends the template's own
  forecasting prompt (question, criteria, public research, answer format) to GUARD.IAN's A2A endpoint through
  `a2a_send.mjs`, which uses upstream's `A2AClient` from @xpr-agents/sdk for the message shape, the signing and the
  no-redirect rule. GUARD.IAN answers with read-only tools only (his runner's A2A tool mode).
- His reply is validated strictly (state, format, range, private markers). Then the template's normal path takes over:
  the parser model only copies the numbers into the library's types, the library aggregates, and (only with --live)
  the library submits one forecast and one comment per question.

What never happens
- No fallback: `GuardIanLlm.invoke` never calls a stock model. On a timeout, a transport error, a decline, a bad
  format or a private marker it raises `GuardIanError`; the library then drops that prediction, and with no valid
  prediction the question is skipped and logged. Nothing is retried, so no forecast is ever re-run because of its value.
- In save-only mode `SaveOnlyMetaculusClient` refuses every Metaculus write before any request is made.
"""

from __future__ import annotations

import asyncio
import contextvars
import ipaddress
import json
import logging
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

logger = logging.getLogger(__name__)

GUARD_IAN_MODEL_NAME = "guard-ian/a2a"  # shown in the public comment's "LLMs" line
REQUEST_HEADER = "GUARD.IAN FORECAST REQUEST v1"  # the brain skill guard-ian-forecaster answers only this header
MAX_REQUEST_CHARS = 500_000  # the brain's express.json limit is 1 MB
A2A_HELPER = Path(__file__).with_name("a2a_send.mjs")
PERCENTILES = (10, 20, 40, 60, 80, 90)
REQUIRED_GUARD_IAN_ENV = ("GUARD_IAN_A2A_ENDPOINT", "GUARD_IAN_ACCOUNT", "A2A_SIGNING_KEY")

# The question being forecast right now. Set by GuardIanQuestionContext so GuardIanLlm knows the question type.
CURRENT_QUESTION: contextvars.ContextVar[MetaculusQuestion | None] = contextvars.ContextVar(
    "guard_ian_current_question", default=None
)


class GuardIanError(Exception):
    """GUARD.IAN's forecast cannot be used. The prediction is dropped; nothing falls back to another model."""


class GuardIanTransportError(GuardIanError):
    """The A2A call failed: timeout, network, HTTP or JSON-RPC error, or the helper could not run."""


class GuardIanDeclined(GuardIanError):
    """GUARD.IAN declined the request (a single 'DECLINED: ...' line)."""


class GuardIanBadReply(GuardIanError):
    """The reply was not a completed task, broke the answer format or range, or carried a private marker."""


# ----------------------------------------------------------------------------- transport

Transport = Callable[[str], Awaitable[dict]]


class NodeA2ATransport:
    """Runs `node a2a_send.mjs` once per request. The helper reads its settings and key from the environment."""

    def __init__(self, helper: Path = A2A_HELPER, node: str = "node", timeout_s: float | None = None) -> None:
        self.helper = helper
        self.node = node
        self.timeout_s = timeout_s or float(os.getenv("GUARD_IAN_A2A_TIMEOUT_S", "300"))

    async def __call__(self, text: str) -> dict:
        node = shutil.which(self.node)
        if node is None:
            raise GuardIanTransportError("node is not installed; run `npm ci` and install Node 18+")
        proc = await asyncio.create_subprocess_exec(
            node,
            str(self.helper),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.helper.parent),
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(json.dumps({"text": text}).encode("utf-8")),
                timeout=self.timeout_s + 30,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise GuardIanTransportError(f"A2A helper did not finish within {self.timeout_s + 30:.0f} s")
        if proc.returncode != 0:
            message = err.decode("utf-8", errors="replace").strip()[:500]
            raise GuardIanTransportError(f"A2A helper exited {proc.returncode}: {message}")
        try:
            task = json.loads(out.decode("utf-8"))
        except ValueError as e:
            raise GuardIanTransportError(f"A2A helper did not print JSON: {e}") from e
        if not isinstance(task, dict):
            raise GuardIanTransportError("A2A helper printed JSON that is not an object")
        return task


# ----------------------------------------------------------------------------- request and reply


def question_type_of(question: MetaculusQuestion) -> str:
    if isinstance(question, BinaryQuestion):
        return "binary"
    if isinstance(question, MultipleChoiceQuestion):
        return "multiple_choice"
    if isinstance(question, NumericQuestion):  # includes discrete questions
        return "numeric"
    if isinstance(question, DateQuestion):
        return "date"
    raise GuardIanBadReply(f"GUARD.IAN is not asked about questions of type {type(question).__name__}")


def build_request(prompt: str, question: MetaculusQuestion) -> str:
    """Wraps the template's forecasting prompt in the header the brain skill expects."""
    request = "\n".join(
        [
            REQUEST_HEADER,
            f"question_type: {question_type_of(question)}",
            "--- BEGIN REQUEST ---",
            prompt.strip(),
            "--- END REQUEST ---",
            "Reply with your public rationale and then the final answer in the strict format for this question_type as "
            "the last line(s), or with one line that starts with DECLINED:.",
        ]
    )
    if len(request) > MAX_REQUEST_CHARS:
        raise GuardIanBadReply(f"request is {len(request)} characters, over the {MAX_REQUEST_CHARS} limit")
    return request


def reply_text_from_task(task: dict) -> str:
    """The text GUARD.IAN returned, from a completed A2A task (artifacts[].parts[] of type text)."""
    state = (task.get("status") or {}).get("state")
    if state != "completed":
        raise GuardIanBadReply(f"A2A task state is {state!r}, not 'completed'")
    texts = [
        part["text"]
        for artifact in task.get("artifacts") or []
        if isinstance(artifact, dict)
        for part in artifact.get("parts") or []
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
    ]
    text = "\n".join(texts).strip()
    if not text:
        raise GuardIanBadReply("A2A task has no text in its artifacts")
    return text


# Markers that must never reach a public comment. Names of wallets, people or memory keys are NOT listed here
# (this repository is public); put them in GUARD_IAN_PRIVATE_TERMS, comma-separated, on the machine that runs the bot.
_PRIVATE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"PVT_K1_", "private key"),
    (r"SIG_K1_", "signature"),
    (r"\b5[HJK][1-9A-HJ-NP-Za-km-z]{49}\b", "WIF private key"),
    (r"\b(team|shared|guard|guard\.ian) notebook\b", "internal notebook"),
    (r"\bguard-ian-(forecaster|persona|sources)\b|\bSKILL\.md\b|\bAGENT_SKILLS\b", "internal skill name"),
    (r"never says safe|gives clarity|before you sign|let me look before|what i checked,? and what i couldn", "banned line"),
    (r"never tell you something is safe", "banned line"),
)


def _private_terms() -> list[str]:
    """Words from GUARD_IAN_PRIVATE_TERMS (comma-separated) plus GUARD_IAN_PRIVATE_TERMS_FILE (default: an untracked
    private_terms.txt next to this file; one term per line, # starts a comment). Never commit real terms."""
    terms = os.getenv("GUARD_IAN_PRIVATE_TERMS", "").split(",")
    path = Path(os.getenv("GUARD_IAN_PRIVATE_TERMS_FILE") or Path(__file__).with_name("private_terms.txt"))
    if path.is_file():
        terms += [line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")]
    return [t.strip() for t in terms if t.strip()]


def find_private_markers(text: str) -> list[str]:
    found = [label for pattern, label in _PRIVATE_PATTERNS if re.search(pattern, text, re.IGNORECASE)]
    found += [
        "private term"
        for term in _private_terms()
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE)
    ]
    return sorted(set(found))


# Extra shapes for the public comment (research text can carry them too): common API keys and PEM private keys.
_SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:sk-ant-|sk-|gsk_|ghp_|gho_|ghs_|ghu_|github_pat_|xox[abposr]-|AIza)[A-Za-z0-9_\-]{20,}", "API key"),
    (r"\bAKIA[0-9A-Z]{16}\b", "API key"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
)
# Anything shaped like an IPv4 or IPv6 address; Python's ipaddress module decides whether it really is one.
_ADDRESS_CANDIDATE = re.compile(
    r"(?<![\w.:])(?:\d{1,3}(?:\.\d{1,3}){3}|[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7})(?![\w:]|\.\d)"
)


def scrub_public_comment(text: str) -> tuple[str, list[str]]:
    """Removes private material from text that will be posted to Metaculus. Returns (clean text, labels removed)."""
    removed: set[str] = set()

    def cut(label: str) -> Callable[[re.Match], str]:
        def replace(_: re.Match) -> str:
            removed.add(label)
            return "[removed]"

        return replace

    for pattern, label in _PRIVATE_PATTERNS + _SECRET_PATTERNS:
        text = re.sub(pattern, cut(label), text, flags=re.IGNORECASE)
    for term in _private_terms():
        text = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", cut("private term"), text, flags=re.IGNORECASE)

    def address(m: re.Match) -> str:
        try:
            ipaddress.ip_address(m.group(0))
        except ValueError:
            return m.group(0)
        return cut("server address")(m)

    return _ADDRESS_CANDIDATE.sub(address, text), sorted(removed)


def _answer_lines(text: str, count: int) -> list[str]:
    lines = [line.strip().strip("*`\"").strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) < count:
        raise GuardIanBadReply(f"reply has fewer than the {count} final answer line(s) required")
    return lines[-count:]


def _percent(value: str, where: str) -> float:
    match = re.fullmatch(r"(\d{1,3}(?:\.\d+)?)\s*%", value.strip())
    if not match:
        raise GuardIanBadReply(f"{where}: {value.strip()!r} is not a percentage like '37%'")
    number = float(match.group(1))
    if not 0 <= number <= 100:
        raise GuardIanBadReply(f"{where}: {number} is outside 0-100")
    return number


def _validate_binary(text: str) -> None:
    (line,) = _answer_lines(text, 1)
    match = re.fullmatch(r"Probability:\s*(.+)", line)
    if not match:
        raise GuardIanBadReply(f"last line {line[:80]!r} is not 'Probability: ZZ%'")
    _percent(match.group(1), "Probability")


def _normalise(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().casefold()


def _validate_multiple_choice(text: str, question: MultipleChoiceQuestion) -> None:
    options = list(question.options)
    lines = _answer_lines(text, len(options))
    total = 0.0
    for option, line in zip(options, lines):
        if ":" not in line:
            raise GuardIanBadReply(f"answer line {line[:80]!r} is not 'Option: NN%'")
        name, value = line.rsplit(":", 1)
        if _normalise(name) != _normalise(option):
            raise GuardIanBadReply(f"answer line {line[:80]!r} does not match option {option!r} in order")
        total += _percent(value, f"option {option!r}")
    if not 99 <= total <= 101:
        raise GuardIanBadReply(f"option probabilities add up to {total}, not 100")


_PERCENTILE_LINE = re.compile(r"Percentile\s+(\d{1,2})\s*:\s*(.+)")
_NUMBER = re.compile(r"[$€£]?(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?![\d.,eE])")
_TRAILING_NOTE = re.compile(r"\s*\([^()]*\)\s*$")  # e.g. "(oldest date)", copied from the template's format text


def _validate_numeric(text: str) -> None:
    values: list[float] = []
    for expected, line in zip(PERCENTILES, _answer_lines(text, len(PERCENTILES))):
        match = _PERCENTILE_LINE.fullmatch(line)
        if not match or int(match.group(1)) != expected:
            raise GuardIanBadReply(f"answer line {line[:80]!r} is not 'Percentile {expected}: <number>'")
        number = _NUMBER.match(match.group(2).strip())
        if not number:
            raise GuardIanBadReply(f"Percentile {expected}: {match.group(2)[:40]!r} is not a plain number")
        value = float(number.group(1).replace(",", ""))
        if not math.isfinite(value):
            raise GuardIanBadReply(f"Percentile {expected} is not finite")
        values.append(value)
    if any(b <= a for a, b in zip(values, values[1:])):
        raise GuardIanBadReply(f"percentile values {values} do not increase")


def _validate_date(text: str) -> None:
    values: list[datetime] = []
    for expected, line in zip(PERCENTILES, _answer_lines(text, len(PERCENTILES))):
        match = _PERCENTILE_LINE.fullmatch(line)
        if not match or int(match.group(1)) != expected:
            raise GuardIanBadReply(f"answer line {line[:80]!r} is not 'Percentile {expected}: YYYY-MM-DD'")
        raw = _TRAILING_NOTE.sub("", match.group(2)).strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?Z)?", raw):
            raise GuardIanBadReply(f"Percentile {expected}: {raw[:40]!r} is not YYYY-MM-DD")
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as e:
            raise GuardIanBadReply(f"Percentile {expected}: {raw!r} is not a real date") from e
        values.append(parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc))
    if any(b <= a for a, b in zip(values, values[1:])):
        raise GuardIanBadReply("percentile dates are not in increasing order")


def validate_reply(text: str, question: MetaculusQuestion) -> None:
    """Raises GuardIanError unless the reply is a usable, public-safe forecast for this question."""
    if re.match(r"\s*DECLINED:", text, re.IGNORECASE):
        raise GuardIanDeclined(text.strip().splitlines()[0][:200])
    markers = find_private_markers(text)
    if markers:
        raise GuardIanBadReply(f"reply carries private markers: {', '.join(markers)}")
    kind = question_type_of(question)
    if kind == "binary":
        _validate_binary(text)
    elif kind == "multiple_choice":
        _validate_multiple_choice(text, question)  # type: ignore[arg-type]
    elif kind == "numeric":
        _validate_numeric(text)
    else:
        _validate_date(text)


# ----------------------------------------------------------------------------- the forecaster


class GuardIanLlm(GeneralLlm):
    """The bot's only forecaster. Looks like a GeneralLlm to the template; never calls a model itself."""

    def __init__(self, transport: Transport | None = None, save_dir: str | None = None) -> None:
        super().__init__(model=GUARD_IAN_MODEL_NAME, allowed_tries=1, temperature=None)
        self._transport: Transport = transport or NodeA2ATransport()
        self._save_dir = save_dir
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _one_at_a_time(self) -> asyncio.Lock:
        # One brain run at a time: the brain allows 5 runs at once, shared with its job work.
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock, self._lock_loop = asyncio.Lock(), loop
        return self._lock

    async def invoke(self, prompt: Any, system_prompt: str | None = None) -> str:
        question = CURRENT_QUESTION.get()
        if question is None:
            raise GuardIanError("GUARD.IAN is asked only through GuardIanForecastBot (no question in context)")
        if not isinstance(prompt, str) or system_prompt is not None:
            raise GuardIanError("GuardIanLlm takes one plain-text prompt")
        request = build_request(prompt, question)
        task: dict | None = None
        try:
            async with self._one_at_a_time():
                task = await self._transport(request)
            text = reply_text_from_task(task)
            validate_reply(text, question)
        except GuardIanError as e:
            logger.warning(f"GUARD.IAN forecast not used for {question.page_url or 'question'}: {e}")
            self._save(question, task, outcome=f"{type(e).__name__}: {e}")
            raise
        self._save(question, task, outcome="ok")
        logger.info(f"GUARD.IAN answered {question.page_url or 'question'} ({len(text)} characters, format ok)")
        return text

    def _save(self, question: MetaculusQuestion, task: dict | None, outcome: str) -> None:
        if not self._save_dir:
            return
        Path(self._save_dir).mkdir(parents=True, exist_ok=True)
        record = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "question_text": question.question_text,
            "outcome": outcome,
            "a2a_task": task,
        }
        with open(Path(self._save_dir) / "guard-ian-a2a-replies.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class GuardIanQuestionContext:
    """Bot mixin: tells GuardIanLlm which question each forecast is for (the template's prompt alone does not)."""

    async def _make_prediction(self, question: MetaculusQuestion, research: str):  # type: ignore[override]
        token = CURRENT_QUESTION.set(question)
        try:
            return await super()._make_prediction(question, research)  # type: ignore[misc]
        finally:
            CURRENT_QUESTION.reset(token)


def assert_guard_ian_is_the_only_forecaster(bot: Any) -> None:
    """Stops the run unless every forecast of this bot goes to GUARD.IAN."""
    forecaster = bot.get_llm("default")
    if not isinstance(forecaster, GuardIanLlm):
        raise SystemExit(f"Refusing to run: the forecaster is {forecaster!r}, not GUARD.IAN")
    if not isinstance(bot, GuardIanQuestionContext):
        raise SystemExit("Refusing to run: the bot does not pass questions to GUARD.IAN")


# ----------------------------------------------------------------------------- save-only mode


class SaveOnlyMetaculusClient(MetaculusClient):
    """Metaculus client for save-only runs: every write method raises before any request is made."""

    WRITE_METHODS = (
        "create_bot",
        "post_question_comment",
        "post_binary_question_prediction",
        "post_numeric_question_prediction",
        "post_multiple_choice_question_prediction",
        "post_question_link",
        "delete_question_link",
        "resolve_question",
        "unresolve_question",
        "create_question",
        "approve_question",
        "_post_question_prediction",
        "_post_bulk_forecasts_and_comments",
    )


def _refuse(name: str) -> Callable[..., Any]:
    def refused(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"save-only mode: Metaculus write '{name}' is disabled")

    return refused


for _name in SaveOnlyMetaculusClient.WRITE_METHODS:
    setattr(SaveOnlyMetaculusClient, _name, _refuse(_name))


def synthetic_question() -> BinaryQuestion:
    """A made-up binary question for the save-only test. It is not on Metaculus and has no post or question id."""
    return BinaryQuestion(
        question_text=(
            "Will the US Bureau of Labor Statistics report November 2026 CPI-U year-over-year inflation at or "
            "above 3.0% in a release published on or before 2026-12-20?"
        ),
        resolution_criteria=(
            "Resolves Yes if the BLS Consumer Price Index news release for November 2026, published on or before "
            "2026-12-20, reports the 12-month percent change in the all-items CPI-U (not seasonally adjusted) as "
            "3.0% or higher. Resolves No if it reports a value below 3.0%, or if no November 2026 CPI release is "
            "published on or before 2026-12-20."
        ),
        fine_print=(
            "Only the headline figure in the first BLS release counts; later revisions do not. This practice "
            "question was written for a save-only test of the bot. It is not a Metaculus question and no forecast "
            "is submitted."
        ),
        background_info=(
            "The BLS publishes the Consumer Price Index each month at https://www.bls.gov/cpi/ . The headline "
            "figure is the 12-month change in the all-items index for all urban consumers (CPI-U)."
        ),
        custom_metadata={"synthetic": True},
    )


def missing_guard_ian_env() -> list[str]:
    return [name for name in REQUIRED_GUARD_IAN_ENV if not os.getenv(name, "").strip()]
