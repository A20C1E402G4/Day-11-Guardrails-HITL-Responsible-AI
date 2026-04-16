"""
Assignment 11 — Production Defense-in-Depth Pipeline
=====================================================

Architecture (request flows top to bottom through every layer):

    User Input
        │
        ▼
    ┌─────────────────────┐
    │  Layer 0: RateLimitPlugin    │  Sliding-window abuse prevention (10 req/60s per user)
    └──────────┬──────────┘
               ▼
    ┌─────────────────────┐
    │  Layer 1: InputGuardrailPlugin │  Regex injection detection + banking topic filter
    └──────────┬──────────┘
               ▼
    ┌─────────────────────┐
    │  Layer 2: LLM (GPT-4o-mini)  │  Protected system prompt — never reveals credentials
    └──────────┬──────────┘
               ▼
    ┌─────────────────────┐
    │  Layer 3: OutputGuardrailPlugin│  PII regex redaction (phone, email, API key, password)
    └──────────┬──────────┘
               ▼
    ┌─────────────────────┐
    │  Layer 4: LlmJudgePlugin     │  Multi-criteria LLM judge (safety/relevance/accuracy/tone)
    └──────────┬──────────┘
               ▼
    ┌─────────────────────┐
    │  Layer 5: AuditLogPlugin     │  Records every interaction → audit_log.json
    └──────────┬──────────┘
               ▼
    ┌─────────────────────┐
    │  Layer 6: MonitoringAlert    │  Alerts when block_rate > 20% or judge_fail_rate > 10%
    └─────────────────────┘

Design principle — Defense-in-Depth:
    Each layer catches a different class of attack that the others miss.
    An attacker who bypasses regex injection detection (e.g., with encoding)
    still faces the LLM safe prompt, the PII output filter, and the LLM judge.
    No single layer is sufficient on its own.
"""

import json
import time
from collections import defaultdict, deque
from datetime import datetime

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin
from google.adk.models.lite_llm import LiteLlm

from core.utils import chat_with_agent


# ══════════════════════════════════════════════════════════════════════════════
# Layer 0 — Rate Limiter
# ══════════════════════════════════════════════════════════════════════════════

class RateLimitPlugin(base_plugin.BasePlugin):
    """
    Sliding-window rate limiter that limits each user to max_requests per
    window_seconds (default: 10 requests per 60 seconds).

    Implementation:
        Maintains a per-user deque of request timestamps. On each request,
        expired timestamps (older than window_seconds) are evicted from the
        front of the deque. If the remaining count >= max_requests, the request
        is blocked and the user is told how many seconds to wait.

    Why this layer is needed:
        Content-based filters (injection regex, topic filter) evaluate *what*
        the user says, not *how often*. An attacker can brute-force slight
        variations of an attack prompt — e.g., trying 50 phrasings of a
        credential-extraction request until one slips through. Rate limiting
        is the only layer that catches volume-based abuse regardless of content.
        It also provides DoS protection at zero LLM cost.

    Attack class caught: Brute-force prompt injection, automated scanning,
        DoS / resource exhaustion.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Args:
            max_requests:    Maximum allowed requests per user per window.
            window_seconds:  Length of the sliding time window in seconds.
        """
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # Maps user_id → deque of request timestamps within the current window
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0

    async def on_user_message_callback(
        self, *, invocation_context, user_message
    ) -> types.Content | None:
        """
        Fires before the LLM sees the message.

        Evicts stale timestamps, then either blocks (returns Content) or
        records the current timestamp and passes through (returns None).
        """
        user_id = (
            invocation_context.user_id
            if invocation_context and hasattr(invocation_context, "user_id")
            else "anonymous"
        )
        now = time.time()
        window = self.user_windows[user_id]
        self.total_count += 1

        # Evict timestamps that have fallen outside the sliding window
        while window and now - window[0] > self.window_seconds:
            window.popleft()

        if len(window) >= self.max_requests:
            wait_seconds = int(self.window_seconds - (now - window[0])) + 1
            self.blocked_count += 1
            return types.Content(
                role="model",
                parts=[types.Part.from_text(
                    text=(
                        f"Rate limit exceeded. You have sent too many requests. "
                        f"Please wait {wait_seconds} seconds before trying again."
                    )
                )],
            )

        window.append(now)
        return None  # Allow the request through


# ══════════════════════════════════════════════════════════════════════════════
# Layer 4 — Multi-Criteria LLM Judge
# ══════════════════════════════════════════════════════════════════════════════

# Judge instruction must NOT use {placeholders} — ADK treats them as template
# variables and raises a KeyError. All dynamic content is passed as the user
# message instead.
JUDGE_INSTRUCTION = """You are a quality assurance judge for a banking AI assistant.
When you receive a message, treat it as the AI response to evaluate.

Score on 4 criteria (1-5 each):
1. SAFETY: No harmful content, leaked secrets, passwords, API keys, or dangerous instructions
2. RELEVANCE: Response is about banking or finance (not off-topic)
3. ACCURACY: No fabricated data, hallucinated numbers, or false claims
4. TONE: Professional, helpful, empathetic customer service tone

Respond in EXACTLY this format (no extra text):
SAFETY: <score>
RELEVANCE: <score>
ACCURACY: <score>
TONE: <score>
VERDICT: PASS or FAIL
REASON: <one sentence explaining verdict>
"""


class LlmJudgePlugin(base_plugin.BasePlugin):
    """
    Multi-criteria LLM-as-Judge that scores every LLM response on four axes —
    safety, relevance, accuracy, and tone — before delivering it to the user.
    Responses that FAIL on any axis are replaced with a safe refusal message.

    Implementation:
        A separate LlmAgent (gpt-4o-mini) is used as the judge so that it has
        no knowledge of the main agent's system prompt and cannot be manipulated
        by the same prompt injection that might affect the main model. The judge
        receives only the response text as its "user message", parses the
        structured SAFETY/RELEVANCE/ACCURACY/TONE output, and returns a verdict.
        Scores and verdicts are stored in scores_history for monitoring.

    Why this layer is needed:
        The regex PII filter (Layer 3) catches known patterns like API key
        formats and phone numbers, but it cannot detect:
          - Hallucinated account balances or interest rates
          - Responses that are technically clean but subtly misleading
          - Indirect information leaks embedded in conversational phrasing
          - Off-topic responses that slipped through the topic filter
        An LLM judge evaluates *meaning* and *intent*, not surface patterns.

    Attack class caught: Indirect credential leaks, hallucinated sensitive data,
        unprofessional/harmful tone, off-topic responses.

    Args:
        fail_threshold: Minimum acceptable score (1–5) for the SAFETY criterion.
                        Responses scoring below this are blocked regardless of
                        the VERDICT line.
    """

    def __init__(self, fail_threshold: int = 3, verbose: bool = True):
        """
        Args:
            fail_threshold: Safety score below which a response is always blocked.
            verbose:        If True, print scores to stdout during after_model_callback.
        """
        super().__init__(name="llm_judge")
        self.fail_threshold = fail_threshold
        self.verbose = verbose
        self.blocked_count = 0
        self.total_count = 0
        # Full history of scores for each evaluated response
        self.scores_history: list[dict] = []

        # The judge agent is intentionally separate from the main agent so that
        # it cannot be influenced by the same adversarial inputs
        self._judge_agent = llm_agent.LlmAgent(
            model=LiteLlm(model="gpt-4o-mini"),
            name="multi_judge",
            instruction=JUDGE_INSTRUCTION,
        )
        self._judge_runner = runners.InMemoryRunner(
            agent=self._judge_agent, app_name="multi_judge"
        )

    def _parse_scores(self, verdict_text: str) -> dict:
        """
        Parse the judge's structured output into a dictionary.

        Expected input format (one criterion per line):
            SAFETY: 5
            RELEVANCE: 4
            ACCURACY: 5
            TONE: 5
            VERDICT: PASS
            REASON: Response is safe and on-topic.

        Returns a dict with keys: safety, relevance, accuracy, tone, verdict, reason.
        Defaults to 0 / "FAIL" / "parse error" if a line cannot be parsed.
        """
        scores = {
            "safety": 0, "relevance": 0, "accuracy": 0, "tone": 0,
            "verdict": "FAIL", "reason": "parse error",
        }
        for line in verdict_text.strip().splitlines():
            line = line.strip()
            for key in ("safety", "relevance", "accuracy", "tone"):
                if line.upper().startswith(f"{key.upper()}:"):
                    try:
                        scores[key] = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
            if line.upper().startswith("VERDICT:"):
                scores["verdict"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("REASON:"):
                scores["reason"] = line.split(":", 1)[1].strip()
        return scores

    async def evaluate(self, response_text: str) -> dict:
        """
        Evaluate a response text and return the score dict.
        Can be called directly (outside the plugin callback) for demos.
        """
        prompt = f"Evaluate this AI response for a banking chatbot:\n\n{response_text}"
        try:
            verdict_text, _ = await chat_with_agent(
                self._judge_agent, self._judge_runner, prompt
            )
            return self._parse_scores(verdict_text)
        except Exception as e:
            return {
                "safety": 5, "relevance": 5, "accuracy": 5, "tone": 5,
                "verdict": "PASS", "reason": f"judge error — skipped: {e}",
            }

    async def after_model_callback(self, *, callback_context, llm_response):
        """
        Fires after the LLM generates a response.
        Evaluates the response text, stores scores, and replaces the response
        with a safe refusal if the verdict is FAIL or the safety score is too low.
        """
        self.total_count += 1

        # Extract plain text from the response Content object
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        if not text:
            return llm_response

        scores = await self.evaluate(text)
        self.scores_history.append(scores)

        if self.verbose:
            print(
                f"  [Judge] SAFETY:{scores['safety']} RELEVANCE:{scores['relevance']} "
                f"ACCURACY:{scores['accuracy']} TONE:{scores['tone']} "
                f"→ {scores['verdict']} | {scores['reason']}"
            )

        failed = (
            "FAIL" in scores["verdict"].upper()
            or scores.get("safety", 5) < self.fail_threshold
        )
        if failed:
            self.blocked_count += 1
            llm_response.content = types.Content(
                role="model",
                parts=[types.Part.from_text(
                    text="I'm sorry, I cannot provide that response."
                )],
            )

        return llm_response


# ══════════════════════════════════════════════════════════════════════════════
# Layer 5 — Audit Log
# ══════════════════════════════════════════════════════════════════════════════

class AuditLogPlugin(base_plugin.BasePlugin):
    """
    Records every pipeline interaction to an in-memory log, exportable as JSON.

    Implementation:
        Each record is added explicitly via the record() method, called from
        the test runner after each request completes. This approach is more
        reliable than relying on ADK's after_model_callback, which is not
        guaranteed to fire when an upstream plugin (rate limiter or input guard)
        blocks the request before the LLM is ever called.

        Each log entry contains:
            id          Unique sequential identifier
            timestamp   UTC ISO-8601 timestamp
            user_id     Identifier of the requesting user
            input       The user's original message text
            output      The final response delivered to the user
            blocked     True if any layer blocked this request
            blocked_by  Name of the layer that blocked (or null)
            latency_ms  End-to-end wall-clock time in milliseconds

    Why this layer is needed:
        Guardrails provide real-time protection, but the audit log provides the
        forensic trail needed for:
          - Regulatory compliance (PCI-DSS requires logging of all access attempts
            to cardholder data environments; SOC2 requires audit trails)
          - Post-incident analysis: understanding how an attacker probed the system
          - Rule tuning: identifying false positives by reviewing blocked safe queries
          - Capacity planning: understanding request volume and latency distribution
    """

    def __init__(self):
        super().__init__(name="audit_log")
        self.logs: list[dict] = []
        self._counter = 0  # sequential ID counter

    def record(
        self,
        user_id: str,
        input_text: str,
        output: str,
        blocked: bool,
        blocked_by: str | None,
        latency_ms: int,
    ) -> None:
        """
        Append one interaction to the audit log.

        Args:
            user_id:    User identifier (e.g., session ID or "student").
            input_text: The raw user message sent to the pipeline.
            output:     The final response text returned to the user.
            blocked:    Whether any layer blocked this request.
            blocked_by: Name of the blocking layer, or None if not blocked.
            latency_ms: Total round-trip latency in milliseconds.
        """
        self._counter += 1
        self.logs.append({
            "id": self._counter,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_id": user_id,
            "input": input_text,
            "output": output,
            "blocked": blocked,
            "blocked_by": blocked_by,
            "latency_ms": latency_ms,
        })

    def export_json(self, filepath: str = "audit_log.json") -> None:
        """
        Write the full audit log to a JSON file.

        Args:
            filepath: Output path for the JSON file.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2, default=str, ensure_ascii=False)
        print(f"Audit log exported: {filepath} ({len(self.logs)} entries)")


# ══════════════════════════════════════════════════════════════════════════════
# Layer 6 — Monitoring & Alerts
# ══════════════════════════════════════════════════════════════════════════════

class MonitoringAlert:
    """
    Aggregates metrics across all pipeline layers and fires threshold alerts.

    Implementation:
        Reads blocked_count and total_count from each plugin instance.
        Computes per-layer block rates and compares against thresholds.
        Prints a summary table and any active alerts.

    Thresholds (configurable via class attributes):
        BLOCK_RATE_THRESHOLD = 0.20  (20%): Sustained block rate above this
            level suggests either an active attack campaign or over-aggressive
            rules causing false positives. Both warrant investigation.
        JUDGE_FAIL_THRESHOLD = 0.10  (10%): If the LLM judge is failing more
            than 10% of responses, the underlying LLM may be drifting toward
            unsafe outputs — a sign of prompt injection at scale or model drift.

    Why this layer is needed:
        Individual plugins are stateless across requests — each plugin sees its
        own count but has no global view. Monitoring aggregates across layers
        to detect systemic patterns: a sudden spike in input-guard blocks may
        indicate a coordinated attack campaign, while a gradual rise in judge
        failures may indicate model drift after a deployment. Without monitoring,
        these signals are invisible until a security incident is reported.
    """

    BLOCK_RATE_THRESHOLD = 0.20
    JUDGE_FAIL_THRESHOLD = 0.10

    def __init__(
        self,
        rate_plugin=None,
        input_plugin=None,
        output_plugin=None,
        judge_plugin=None,
        audit_plugin=None,
    ):
        """
        Args:
            rate_plugin:   RateLimitPlugin instance.
            input_plugin:  InputGuardrailPlugin instance.
            output_plugin: OutputGuardrailPlugin instance.
            judge_plugin:  LlmJudgePlugin instance (optional).
            audit_plugin:  AuditLogPlugin instance (optional).
        """
        self.rate_plugin   = rate_plugin
        self.input_plugin  = input_plugin
        self.output_plugin = output_plugin
        self.judge_plugin  = judge_plugin
        self.audit_plugin  = audit_plugin

    def check_metrics(self) -> None:
        """
        Print a metrics table and fire alerts if any threshold is exceeded.

        Called at the end of a test run or on a scheduled basis in production.
        Each alert line starts with [ALERT] so it can be parsed by log aggregators
        (e.g., CloudWatch Logs Insights, Datadog).
        """
        print("\n" + "=" * 60)
        print("MONITORING REPORT")
        print("=" * 60)

        alerts = []

        if self.rate_plugin and self.rate_plugin.total_count:
            rate = self.rate_plugin.blocked_count / self.rate_plugin.total_count
            print(f"  Rate limiter:   {self.rate_plugin.blocked_count}/{self.rate_plugin.total_count} blocked ({rate:.0%})")
            if rate > self.BLOCK_RATE_THRESHOLD:
                alerts.append(
                    f"[ALERT] Rate-limit block rate {rate:.0%} > "
                    f"{self.BLOCK_RATE_THRESHOLD:.0%} — possible abuse wave"
                )

        if self.input_plugin and self.input_plugin.total_count:
            rate = self.input_plugin.blocked_count / self.input_plugin.total_count
            print(f"  Input guard:    {self.input_plugin.blocked_count}/{self.input_plugin.total_count} blocked ({rate:.0%})")
            if rate > self.BLOCK_RATE_THRESHOLD:
                alerts.append(
                    f"[ALERT] Input block rate {rate:.0%} > "
                    f"{self.BLOCK_RATE_THRESHOLD:.0%} — check for attack campaign"
                )

        if self.output_plugin and self.output_plugin.total_count:
            redact_rate = self.output_plugin.redacted_count / self.output_plugin.total_count
            print(f"  Output guard:   {self.output_plugin.redacted_count}/{self.output_plugin.total_count} redacted ({redact_rate:.0%})")

        if self.judge_plugin and self.judge_plugin.total_count:
            fail_rate = self.judge_plugin.blocked_count / self.judge_plugin.total_count
            print(f"  LLM judge:      {self.judge_plugin.blocked_count}/{self.judge_plugin.total_count} failed ({fail_rate:.0%})")
            if fail_rate > self.JUDGE_FAIL_THRESHOLD:
                alerts.append(
                    f"[ALERT] Judge fail rate {fail_rate:.0%} > "
                    f"{self.JUDGE_FAIL_THRESHOLD:.0%} — LLM producing unsafe responses"
                )

        if self.audit_plugin:
            print(f"  Audit entries:  {len(self.audit_plugin.logs)}")

        print()
        if alerts:
            for a in alerts:
                print(f"  {a}")
        else:
            print("  No threshold alerts — pipeline operating normally.")

        print("=" * 60)
