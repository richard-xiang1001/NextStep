"""Baseline methods for evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..env.base import Action, BaseEnvironment, Observation
from ..env.webshop import create_webshop_env
from ..utils.config import load_config
from ..utils.logging import create_run_manifest
from ..utils.trajectory import Trajectory, log_trajectory


ACTION_SYSTEM_PROMPT = (
    "You are controlling a WebShop text environment. "
    "Return exactly one JSON object with keys: action, args. "
    "Allowed actions are search, click, finish. "
    "Format examples: "
    '{"action":"search","args":{"query":"water bottle 32oz"}} '
    '{"action":"click","args":{"element_key":"Buy Now"}} '
    '{"action":"finish","args":{}} '
    "No markdown, no extra text."
)


@dataclass
class BaselineConfig:
    """Configuration for baseline methods."""

    method: str  # "greedy", "sampling", "rm_bon", "rm_mcts"
    seed: int = 42
    max_steps: int = 80
    max_tokens: int = 6000
    max_calls: int = 12
    num_samples: int = 1
    temperature: float = 0.7
    llm_provider: str = "openai"  # openai | anthropic
    llm_model: str = "gpt-4.1-mini"
    llm_max_output_tokens: int = 128
    llm_top_p: Optional[float] = None
    llm_stop: Optional[str] = None
    llm_reasoning_effort: Optional[str] = None
    llm_tools_enabled: bool = False
    max_obs_chars: int = 2200
    max_trace_obs_chars: int = 1200
    repair_temperature: float = 0.0
    mcts_num_simulations: int = 10
    mcts_branching: int = 3
    mcts_exploration: float = 1.4


@dataclass
class BudgetState:
    """Shared budget envelope for tok/call/step accounting."""

    tok: int = 0
    call: int = 0
    step: int = 0
    max_tok: int = 6000
    max_call: int = 12
    max_step: int = 80
    flags: Dict[str, Any] = field(default_factory=dict)

    def remaining_tokens(self) -> int:
        return max(0, self.max_tok - self.tok)

    def remaining_calls(self) -> int:
        return max(0, self.max_call - self.call)

    def remaining_steps(self) -> int:
        return max(0, self.max_step - self.step)

    def exceeded(self) -> bool:
        return self.tok >= self.max_tok or self.call >= self.max_call or self.step >= self.max_step

    def consume_call(self, used_tok: int, source: str = "policy") -> None:
        self.call += 1
        self.tok += max(1, int(used_tok))
        self.flags[f"calls_{source}"] = int(self.flags.get(f"calls_{source}", 0)) + 1
        if self.call >= self.max_call:
            self.flags["budget_exceeded_call"] = True
        if self.tok >= self.max_tok:
            self.flags["budget_exceeded_tok"] = True

    def consume_step(self, source: str = "env") -> None:
        self.step += 1
        self.flags[f"steps_{source}"] = int(self.flags.get(f"steps_{source}", 0)) + 1
        if self.step >= self.max_step:
            self.flags["budget_exceeded_step"] = True

    def snapshot(self) -> Dict[str, Any]:
        return {
            "tok": self.tok,
            "call": self.call,
            "step": self.step,
            "remaining_tok": self.remaining_tokens(),
            "remaining_call": self.remaining_calls(),
            "remaining_step": self.remaining_steps(),
            "max_tok": self.max_tok,
            "max_call": self.max_call,
            "max_step": self.max_step,
            "exceeded": self.exceeded(),
        }

    def clone(self) -> "BudgetState":
        return BudgetState(
            tok=self.tok,
            call=self.call,
            step=self.step,
            max_tok=self.max_tok,
            max_call=self.max_call,
            max_step=self.max_step,
            flags=dict(self.flags),
        )


@dataclass
class LLMResult:
    """Output of an LLM action call."""

    text: str
    usage_tokens: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    request: Dict[str, Any] = field(default_factory=dict)
    response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionDecision:
    """Decision packet for one environment action."""

    action: Action
    parse_fail: bool
    recovered_by_repair: bool
    used_tokens: int
    llm_calls: int
    llm_text: str
    parsed: Optional[Dict[str, Any]]
    attempts: List[Dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    fallback_used: bool
    fallback_reason: Optional[str]


@dataclass
class MCTSNode:
    """Open-loop MCTS node storing only action prefix."""

    prefix: List[Action]
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[str, "MCTSNode"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.value_sum / self.visits


class LLMPolicyClient:
    """Thin wrapper over OpenAI/Anthropic chat calls for action generation."""

    def __init__(self, config: BaselineConfig):
        self.provider = config.llm_provider.lower()
        self.model = config.llm_model
        self.max_output_tokens = config.llm_max_output_tokens
        self.top_p = config.llm_top_p
        self.stop = config.llm_stop
        self.reasoning_effort = config.llm_reasoning_effort
        self.tools_enabled = config.llm_tools_enabled
        self.openai_timeout_sec = float(os.getenv("OPENAI_TIMEOUT_SEC", "60"))
        self.openai_max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
        self._openai_client = None
        self._anthropic_client = None

        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for greedy/sampling baselines.")
            from openai import OpenAI

            base_url = os.getenv("OPENAI_BASE_URL")
            self._openai_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.openai_timeout_sec,
                max_retries=self.openai_max_retries,
            )
            return

        if self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required when llm_provider=anthropic.")
            import anthropic

            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
            return

        if self.provider == "fake":
            return

        raise ValueError(f"Unsupported llm_provider={config.llm_provider}. Use openai, anthropic, or fake.")

    def _request_anchor(self, temperature: float, seed: Optional[int]) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_p": self.top_p,
            "stop": self.stop,
            "reasoning_effort": self.reasoning_effort,
            "tools_enabled": self.tools_enabled,
            "seed": seed,
        }

    @staticmethod
    def _model_error_result(prompt: str, request_anchor: Dict[str, Any], error: Exception, latency_ms: float) -> LLMResult:
        # Fail-closed action so transient provider outages do not kill full experiment runs.
        fallback_text = json.dumps({"action": "finish", "args": {}}, ensure_ascii=True)
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(fallback_text) // 4)
        return LLMResult(
            text=fallback_text,
            usage_tokens=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=max(0.0, latency_ms),
            request=request_anchor,
            response={
                "provider_error": True,
                "error_type": type(error).__name__,
                "error": str(error),
                "fallback_action": "finish",
            },
        )

    def generate(self, prompt: str, temperature: float, seed: Optional[int] = None) -> LLMResult:
        """Generate one action response from the configured provider."""
        request_anchor = self._request_anchor(temperature=temperature, seed=seed)
        if self.provider == "fake":
            return self._generate_fake(prompt=prompt, seed=seed, request_anchor=request_anchor)

        if self.provider == "openai":
            assert self._openai_client is not None
            kwargs: Dict[str, Any] = {}
            if seed is not None:
                kwargs["seed"] = int(seed)
            if self.top_p is not None:
                kwargs["top_p"] = float(self.top_p)
            if self.stop:
                kwargs["stop"] = self.stop
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
            start = time.perf_counter()
            try:
                resp = self._openai_client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=self.max_output_tokens,
                    messages=[
                        {"role": "system", "content": ACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    **kwargs,
                )
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000.0
                return self._model_error_result(prompt=prompt, request_anchor=request_anchor, error=e, latency_ms=latency_ms)
            latency_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
            content = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            if total_tokens <= 0:
                total_tokens = max(1, (len(prompt) + len(content)) // 4)
            if prompt_tokens <= 0:
                prompt_tokens = max(1, len(prompt) // 4)
            if completion_tokens <= 0:
                completion_tokens = max(1, len(content) // 4)
            return LLMResult(
                text=content,
                usage_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                request=request_anchor,
                response={
                    "id": getattr(resp, "id", None),
                    "finish_reason": getattr(resp.choices[0], "finish_reason", None),
                },
            )

        assert self._anthropic_client is not None
        kwargs = {}
        if self.top_p is not None:
            kwargs["top_p"] = float(self.top_p)
        if self.stop:
            kwargs["stop_sequences"] = [self.stop]
        start = time.perf_counter()
        try:
            resp = self._anthropic_client.messages.create(
                model=self.model,
                system=ACTION_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=self.max_output_tokens,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return self._model_error_result(prompt=prompt, request_anchor=request_anchor, error=e, latency_ms=latency_ms)
        latency_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
        content_parts = getattr(resp, "content", []) or []
        text = "".join(getattr(part, "text", "") for part in content_parts)
        usage = getattr(resp, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = input_tokens + output_tokens
        if total_tokens <= 0:
            total_tokens = max(1, (len(prompt) + len(text)) // 4)
        if input_tokens <= 0:
            input_tokens = max(1, len(prompt) // 4)
        if output_tokens <= 0:
            output_tokens = max(1, len(text) // 4)
        return LLMResult(
            text=text,
            usage_tokens=total_tokens,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            latency_ms=latency_ms,
            request=request_anchor,
            response={
                "id": getattr(resp, "id", None),
                "stop_reason": getattr(resp, "stop_reason", None),
            },
        )

    @staticmethod
    def _extract_candidates_from_prompt(prompt: str) -> List[str]:
        for marker in ("Likely clickable candidates:", "Clickable candidates:"):
            idx = prompt.find(marker)
            if idx < 0:
                continue
            content = prompt[idx + len(marker) :].strip()
            first_line = content.splitlines()[0] if content else ""
            try:
                parsed = json.loads(first_line)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if isinstance(x, (str, int, float))]
            except Exception:
                return []
        return []

    def _generate_fake(self, prompt: str, seed: Optional[int], request_anchor: Dict[str, Any]) -> LLMResult:
        rng = random.Random(0 if seed is None else int(seed))
        step_match = re.search(r"Step:\s*(\d+)", prompt)
        step = int(step_match.group(1)) if step_match else 0
        is_repair_prompt = "Invalid response to repair:" in prompt

        force_invalid_step0 = os.getenv("NEXTSTEP_FAKE_FORCE_STEP0_INVALID", "0").strip() in {"1", "true", "TRUE"}
        if force_invalid_step0 and not is_repair_prompt and step == 0:
            text = "INVALID_OUTPUT"
            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(text) // 4)
            return LLMResult(
                text=text,
                usage_tokens=prompt_tokens + completion_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=0.0,
                request=request_anchor,
                response={"provider": "fake", "forced_invalid": True},
            )

        invalid_rate_str = os.getenv("NEXTSTEP_FAKE_INVALID_RATE", "0.0").strip() or "0.0"
        try:
            invalid_rate = max(0.0, min(1.0, float(invalid_rate_str)))
        except ValueError:
            invalid_rate = 0.0

        if not is_repair_prompt and invalid_rate > 0 and rng.random() < invalid_rate:
            text = "INVALID_OUTPUT"
            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(text) // 4)
            return LLMResult(
                text=text,
                usage_tokens=prompt_tokens + completion_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=0.0,
                request=request_anchor,
                response={"provider": "fake", "sampled_invalid": True},
            )

        query_match = re.search(r"Task instruction:\s*(.+)", prompt)
        query = query_match.group(1).strip() if query_match else "product"
        query = " ".join(query.split())[:120]

        candidates = self._extract_candidates_from_prompt(prompt)
        candidates_norm = [str(c).strip() for c in candidates if str(c).strip()]

        action_text = ""
        buy_now = next((c for c in candidates_norm if c.casefold() == "buy now"), None)
        if buy_now is not None:
            action_text = json.dumps({"action": "click", "args": {"element_key": buy_now}}, ensure_ascii=True)
        elif step <= 0:
            action_text = json.dumps({"action": "search", "args": {"query": query}}, ensure_ascii=True)
        else:
            asin = next((c for c in candidates_norm if re.fullmatch(r"[A-Z0-9]{10}", c) is not None), None)
            if asin is not None:
                action_text = json.dumps({"action": "click", "args": {"element_key": asin}}, ensure_ascii=True)
            elif candidates_norm:
                action_text = json.dumps(
                    {"action": "click", "args": {"element_key": candidates_norm[0]}},
                    ensure_ascii=True,
                )
            else:
                action_text = json.dumps({"action": "finish", "args": {}}, ensure_ascii=True)

        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(action_text) // 4)
        return LLMResult(
            text=action_text,
            usage_tokens=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=0.0,
            request=request_anchor,
            response={"provider": "fake"},
        )


class RMScorer:
    """Week-1 RM scorer stub; replace with RM checkpoint inference in Week-2."""

    def __init__(self, rm_checkpoint: Optional[str] = None):
        self.rm_checkpoint = rm_checkpoint

    def score(self, trajectory: Trajectory) -> float:
        score = 0.0
        if trajectory.env_success:
            score += 1.0
        if trajectory.flags and trajectory.flags.get("parse_fail"):
            score -= 0.1
        if trajectory.flags and trajectory.flags.get("budget_exceeded"):
            score -= 0.2
        cost_tok = float((trajectory.cost or {}).get("tok", 0.0))
        score -= 1e-5 * cost_tok
        return score


class BaselineAgent:
    """Base class for baseline agents."""

    def __init__(self, config: BaselineConfig, env_config: Dict[str, Any]):
        self.config = config
        self.env_config = env_config
        self.env: Optional[BaseEnvironment] = None
        self.rng = random.Random(config.seed)

    def _new_budget_state(self) -> BudgetState:
        return BudgetState(
            max_tok=self.config.max_tokens,
            max_call=self.config.max_calls,
            max_step=self.config.max_steps,
        )

    @staticmethod
    def _stable_text_hash(text: str) -> int:
        h = 2166136261
        for ch in text:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return h & 0xFFFFFFFF

    @classmethod
    def derive_task_seed(cls, base_seed: int, task_id: Optional[str]) -> int:
        token = "unknown" if task_id is None else str(task_id)
        return (base_seed * 1315423911 + cls._stable_text_hash(token)) & 0xFFFFFFFF

    @staticmethod
    def derive_child_seed(parent_seed: int, child_idx: int) -> int:
        return ((parent_seed + 0x9E3779B9) ^ (child_idx * 2654435761)) & 0xFFFFFFFF

    @staticmethod
    def _normalize_search_query(query: str) -> str:
        query = " ".join(query.strip().split())
        return query[:160] if len(query) > 160 else query

    @staticmethod
    def _extract_instruction_from_obs(obs_text: str) -> str:
        parts = [p.strip() for p in obs_text.split("[SEP]")]
        for part in parts:
            if part.lower().startswith("instruction:"):
                return part.split(":", 1)[-1].strip()
        return ""

    @staticmethod
    def _extract_click_candidates(obs_text: str) -> List[str]:
        segments = [seg.strip() for seg in obs_text.split("[SEP]")]
        candidates: List[str] = []
        for seg in segments:
            if not seg:
                continue
            seg_lower = seg.lower()
            if seg_lower == "webshop":
                continue
            if seg_lower.startswith("instruction:"):
                continue
            if seg_lower.startswith("page ") and "total results" in seg_lower:
                continue
            if seg_lower.startswith("price:") or seg_lower.startswith("rating:"):
                continue
            if seg_lower.startswith("thank you for shopping"):
                continue

            is_asin = re.fullmatch(r"[A-Z0-9]{10}", seg) is not None
            is_nav = seg in {"Back to Search", "Next >", "< Prev", "Description", "Features", "Reviews", "Buy Now"}
            is_short_candidate = len(seg) <= 42 and ":" not in seg
            if is_asin or is_nav or is_short_candidate:
                candidates.append(seg)

        uniq: List[str] = []
        seen = set()
        for c in candidates:
            key = c.casefold()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        return uniq[:40]

    def _fallback_action(
        self,
        obs: Observation,
        query: str,
        step: int,
        click_candidates: List[str],
    ) -> Action:
        obs_lower = obs.text.lower()
        if step == 0:
            return Action(action_type="search", args={"query": self._normalize_search_query(query)})

        buy_candidate = next((c for c in click_candidates if c.casefold() == "buy now"), None)
        if buy_candidate:
            return Action(action_type="click", args={"element_key": buy_candidate})

        asin_candidate = next((c for c in click_candidates if re.fullmatch(r"[A-Z0-9]{10}", c) is not None), None)
        if asin_candidate:
            return Action(action_type="click", args={"element_key": asin_candidate})

        if "back to search" in obs_lower and step <= 2:
            return Action(action_type="search", args={"query": self._normalize_search_query(query)})

        return Action(action_type="finish", args={})

    def run_episode(
        self,
        task_id: Optional[str] = None,
        query: Optional[str] = None,
        split_id: str = "eval",
        episode_seed: Optional[int] = None,
        budget: Optional[BudgetState] = None,
        prefix_actions: Optional[List[Action]] = None,
        max_policy_steps: Optional[int] = None,
        close_env: bool = False,
    ) -> Trajectory:
        raise NotImplementedError


class LLMBasedAgent(BaselineAgent):
    """Common logic for LLM-driven single-agent baselines."""

    def __init__(self, config: BaselineConfig, env_config: Dict[str, Any]):
        super().__init__(config, env_config)
        self.llm = LLMPolicyClient(config)

    def _temperature(self) -> float:
        raise NotImplementedError

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        payload = text.strip()
        if not payload:
            return None
        try:
            parsed = json.loads(payload)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _normalize_action(
        self,
        parsed: Optional[Dict[str, Any]],
        query: str,
        click_candidates: List[str],
    ) -> Optional[Action]:
        if parsed is None:
            return None
        action_name = str(parsed.get("action", "")).strip().lower()
        args = parsed.get("args", {})
        if not isinstance(args, dict):
            args = {}

        if action_name == "search":
            raw_query = str(args.get("query") or parsed.get("query") or query).strip()
            if not raw_query:
                raw_query = query
            return Action(action_type="search", args={"query": self._normalize_search_query(raw_query)})

        if action_name == "click":
            raw_key = str(args.get("element_key") or args.get("element") or parsed.get("element_key") or "").strip()
            if not raw_key:
                return None
            mapping = {candidate.casefold(): candidate for candidate in click_candidates}
            candidate = mapping.get(raw_key.casefold(), raw_key)
            return Action(action_type="click", args={"element_key": candidate})

        if action_name == "finish":
            return Action(action_type="finish", args={})
        return None

    def _build_prompt(
        self,
        obs: Observation,
        query: str,
        step: int,
        budget: BudgetState,
        click_candidates: List[str],
    ) -> str:
        obs_text = obs.text
        if len(obs_text) > self.config.max_obs_chars:
            obs_text = obs_text[: self.config.max_obs_chars - 3] + "..."

        candidates_json = json.dumps(click_candidates[:25], ensure_ascii=True)
        return (
            f"Task instruction: {query}\n"
            f"Step: {step}/{self.config.max_steps}\n"
            f"Budget used: tokens={budget.tok}/{budget.max_tok}, calls={budget.call}/{budget.max_call}\n"
            "Allowed actions:\n"
            '- search: {"action":"search","args":{"query":"..."}}\n'
            '- click: {"action":"click","args":{"element_key":"..."}}\n'
            '- finish: {"action":"finish","args":{}}\n'
            f"Likely clickable candidates: {candidates_json}\n"
            f"Observation:\n{obs_text}\n"
            "Choose the single best next action."
        )

    def _build_repair_prompt(
        self,
        query: str,
        step: int,
        click_candidates: List[str],
        bad_text: str,
    ) -> str:
        bad_text = bad_text.strip()
        if len(bad_text) > 600:
            bad_text = bad_text[:597] + "..."
        candidates_json = json.dumps(click_candidates[:25], ensure_ascii=True)
        return (
            "Your previous response was invalid.\n"
            "Return ONLY valid JSON.\n"
            f"Task instruction: {query}\n"
            f"Step: {step}\n"
            "Valid schemas:\n"
            '{"action":"search","args":{"query":"..."}}\n'
            '{"action":"click","args":{"element_key":"..."}}\n'
            '{"action":"finish","args":{}}\n'
            f"Clickable candidates: {candidates_json}\n"
            f"Invalid response to repair: {bad_text}\n"
            "Output one valid JSON object only."
        )

    def _query_model(
        self,
        prompt: str,
        temperature: float,
        budget: BudgetState,
        seed: Optional[int],
        source: str,
    ) -> Optional[LLMResult]:
        if budget.remaining_calls() <= 0:
            budget.flags["budget_blocked_model_call"] = True
            return None
        if budget.remaining_tokens() <= 0:
            budget.flags["budget_blocked_model_token"] = True
            return None

        result = self.llm.generate(prompt=prompt, temperature=temperature, seed=seed)
        budget.consume_call(result.usage_tokens, source=source)
        return result

    def _step_seed(self, episode_seed: Optional[int], step: int, attempt: int) -> Optional[int]:
        if episode_seed is None:
            return None
        return ((episode_seed * 1103515245 + step * 12345 + attempt * 2246822519) & 0x7FFFFFFF)

    def _decide_action(
        self,
        obs: Observation,
        query: str,
        step: int,
        budget: BudgetState,
        episode_seed: Optional[int],
    ) -> ActionDecision:
        click_candidates = self._extract_click_candidates(obs.text)
        attempts: List[Dict[str, Any]] = []
        total_tokens = 0
        total_calls = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency_ms = 0.0

        prompt = self._build_prompt(obs=obs, query=query, step=step, budget=budget, click_candidates=click_candidates)
        first = self._query_model(
            prompt=prompt,
            temperature=self._temperature(),
            budget=budget,
            seed=self._step_seed(episode_seed, step, 0),
            source="policy",
        )
        if first is None:
            fallback = self._fallback_action(obs, query, step, click_candidates)
            return ActionDecision(
                action=fallback,
                parse_fail=False,
                recovered_by_repair=False,
                used_tokens=0,
                llm_calls=0,
                llm_text="",
                parsed=None,
                attempts=[{"type": "policy", "status": "blocked_by_budget"}],
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0.0,
                fallback_used=True,
                fallback_reason="budget_blocked",
            )

        total_tokens += first.usage_tokens
        total_calls += 1
        total_prompt_tokens += first.prompt_tokens
        total_completion_tokens += first.completion_tokens
        total_latency_ms += first.latency_ms
        parsed_first = self._extract_json(first.text)
        action_first = self._normalize_action(parsed_first, query, click_candidates)
        attempts.append(
            {
                "type": "policy",
                "status": "ok" if action_first is not None else "invalid",
                "used_tokens": first.usage_tokens,
                "usage": {
                    "prompt_tokens": first.prompt_tokens,
                    "completion_tokens": first.completion_tokens,
                    "total_tokens": first.usage_tokens,
                },
                "latency_ms": first.latency_ms,
                "request": first.request,
                "response": first.response,
                "llm_text": first.text,
                "parsed": parsed_first,
            }
        )
        if action_first is not None:
            return ActionDecision(
                action=action_first,
                parse_fail=False,
                recovered_by_repair=False,
                used_tokens=total_tokens,
                llm_calls=total_calls,
                llm_text=first.text,
                parsed=parsed_first,
                attempts=attempts,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                latency_ms=total_latency_ms,
                fallback_used=False,
                fallback_reason=None,
            )

        repair_prompt = self._build_repair_prompt(
            query=query,
            step=step,
            click_candidates=click_candidates,
            bad_text=first.text,
        )
        repair = self._query_model(
            prompt=repair_prompt,
            temperature=self.config.repair_temperature,
            budget=budget,
            seed=self._step_seed(episode_seed, step, 1),
            source="repair",
        )
        if repair is not None:
            total_tokens += repair.usage_tokens
            total_calls += 1
            total_prompt_tokens += repair.prompt_tokens
            total_completion_tokens += repair.completion_tokens
            total_latency_ms += repair.latency_ms
            parsed_repair = self._extract_json(repair.text)
            action_repair = self._normalize_action(parsed_repair, query, click_candidates)
            attempts.append(
                {
                    "type": "repair",
                    "status": "ok" if action_repair is not None else "invalid",
                    "used_tokens": repair.usage_tokens,
                    "usage": {
                        "prompt_tokens": repair.prompt_tokens,
                        "completion_tokens": repair.completion_tokens,
                        "total_tokens": repair.usage_tokens,
                    },
                    "latency_ms": repair.latency_ms,
                    "request": repair.request,
                    "response": repair.response,
                    "llm_text": repair.text,
                    "parsed": parsed_repair,
                }
            )
            if action_repair is not None:
                return ActionDecision(
                    action=action_repair,
                    parse_fail=False,
                    recovered_by_repair=True,
                    used_tokens=total_tokens,
                    llm_calls=total_calls,
                    llm_text=repair.text,
                    parsed=parsed_repair,
                    attempts=attempts,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    latency_ms=total_latency_ms,
                    fallback_used=False,
                    fallback_reason=None,
                )
            last_text = repair.text
            last_parsed = parsed_repair
            fallback_reason = "parse_fail_after_repair"
        else:
            attempts.append({"type": "repair", "status": "blocked_by_budget"})
            last_text = first.text
            last_parsed = parsed_first
            fallback_reason = "parse_fail_repair_budget_blocked"

        fallback = self._fallback_action(obs, query, step, click_candidates)
        return ActionDecision(
            action=fallback,
            parse_fail=True,
            recovered_by_repair=False,
            used_tokens=total_tokens,
            llm_calls=total_calls,
            llm_text=last_text,
            parsed=last_parsed,
            attempts=attempts,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            latency_ms=total_latency_ms,
            fallback_used=True,
            fallback_reason=fallback_reason,
        )

    def _truncate_trace_obs(self, obs: Observation) -> str:
        text = obs.text if obs and obs.text is not None else ""
        if len(text) > self.config.max_trace_obs_chars:
            return text[: self.config.max_trace_obs_chars - 3] + "..."
        return text

    def _compress_metadata(self, metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if metadata is None:
            return None
        try:
            raw = json.dumps(metadata, ensure_ascii=True)
        except Exception:
            return {"unserializable_metadata": True}
        if len(raw) <= 600:
            return metadata
        return {"truncated_json": raw[:597] + "..."}

    def run_episode(
        self,
        task_id: Optional[str] = None,
        query: Optional[str] = None,
        split_id: str = "eval",
        episode_seed: Optional[int] = None,
        budget: Optional[BudgetState] = None,
        prefix_actions: Optional[List[Action]] = None,
        max_policy_steps: Optional[int] = None,
        close_env: bool = False,
    ) -> Trajectory:
        if self.env is None:
            self.env = create_webshop_env(self.env_config)

        budget_state = budget if budget is not None else self._new_budget_state()
        obs, _ = self.env.reset(task_id)
        query_text = query or self.env.get_task_query() or self._extract_instruction_from_obs(obs.text)
        query_text = query_text.strip() or "Find a relevant product."

        trajectory = Trajectory(
            qid=task_id or "unknown",
            query_text=query_text,
            split_id=split_id,
            workflow=[],
            intermediate_outputs=[],
            env_trace=[],
            flags={},
        )

        parse_fail_any = False
        policy_steps = 0
        done = False
        repair_attempts = 0
        repair_successes = 0
        fallback_count = 0
        fallback_budget_count = 0
        fallback_parse_count = 0
        model_prompt_tokens = 0
        model_completion_tokens = 0
        model_latency_ms_total = 0.0
        model_call_count = 0

        replay_actions = prefix_actions or []
        for replay_action in replay_actions:
            if done:
                break
            if budget_state.remaining_steps() <= 0:
                budget_state.flags["budget_exceeded_step"] = True
                break
            sr = self.env.step(replay_action)
            budget_state.consume_step(source="replay")
            trajectory.env_trace.append(
                {
                    "obs": self._truncate_trace_obs(sr.observation),
                    "obs_metadata": self._compress_metadata(sr.observation.metadata),
                    "action": replay_action.to_string(),
                    "replay": True,
                }
            )
            done = sr.done
            obs = sr.observation

        while not done:
            if budget_state.remaining_steps() <= 0:
                budget_state.flags["budget_exceeded_step"] = True
                break
            if max_policy_steps is not None and policy_steps >= max_policy_steps:
                break

            budget_before = budget_state.snapshot()
            decision = self._decide_action(
                obs=obs,
                query=query_text,
                step=policy_steps,
                budget=budget_state,
                episode_seed=episode_seed,
            )
            parse_fail_any = parse_fail_any or decision.parse_fail
            model_prompt_tokens += int(decision.prompt_tokens)
            model_completion_tokens += int(decision.completion_tokens)
            model_latency_ms_total += float(decision.latency_ms)
            model_call_count += int(decision.llm_calls)
            if any(attempt.get("type") == "repair" for attempt in decision.attempts):
                repair_attempts += 1
            if decision.recovered_by_repair:
                repair_successes += 1
            if decision.fallback_used:
                fallback_count += 1
                if decision.fallback_reason and "budget" in decision.fallback_reason:
                    fallback_budget_count += 1
                else:
                    fallback_parse_count += 1

            sr = self.env.step(decision.action)
            budget_state.consume_step(source="env")
            budget_after = budget_state.snapshot()

            trajectory.env_trace.append(
                {
                    "obs": self._truncate_trace_obs(sr.observation),
                    "obs_metadata": self._compress_metadata(sr.observation.metadata),
                    "action": decision.action.to_string(),
                    "replay": False,
                }
            )
            trajectory.intermediate_outputs.append(
                {
                    "t": policy_steps,
                    "llm_text": decision.llm_text,
                    "parsed": decision.parsed,
                    "parse_fail": decision.parse_fail,
                    "recovered_by_repair": decision.recovered_by_repair,
                    "used_tokens": decision.used_tokens,
                    "llm_calls": decision.llm_calls,
                    "attempts": decision.attempts,
                    "prompt_tokens": decision.prompt_tokens,
                    "completion_tokens": decision.completion_tokens,
                    "latency_ms": decision.latency_ms,
                    "fallback_used": decision.fallback_used,
                    "fallback_reason": decision.fallback_reason,
                    "budget_before": budget_before,
                    "budget_after": budget_after,
                }
            )

            done = sr.done
            obs = sr.observation
            policy_steps += 1
            if budget_state.exceeded():
                done = True

        trajectory.env_success = self.env.get_success()
        trajectory.cost = {
            "tok": float(budget_state.tok),
            "step": float(budget_state.step),
            "call": float(budget_state.call),
        }
        trajectory.flags.update(budget_state.flags)
        trajectory.flags["parse_fail"] = parse_fail_any
        trajectory.flags["budget_exceeded"] = budget_state.exceeded()
        trajectory.flags["repair_attempts"] = repair_attempts
        trajectory.flags["repair_successes"] = repair_successes
        trajectory.flags["fallback_count"] = fallback_count
        trajectory.flags["fallback_budget_count"] = fallback_budget_count
        trajectory.flags["fallback_parse_count"] = fallback_parse_count
        trajectory.flags["model_prompt_tokens"] = model_prompt_tokens
        trajectory.flags["model_completion_tokens"] = model_completion_tokens
        trajectory.flags["model_total_tokens"] = model_prompt_tokens + model_completion_tokens
        trajectory.flags["model_call_count"] = model_call_count
        trajectory.flags["model_latency_ms_total"] = round(model_latency_ms_total, 3)
        trajectory.flags["model_latency_ms_avg"] = (
            round(model_latency_ms_total / model_call_count, 3) if model_call_count > 0 else 0.0
        )
        trajectory.flags["budget_cut_tok"] = bool(
            budget_state.flags.get("budget_exceeded_tok") or budget_state.flags.get("budget_blocked_model_token")
        )
        trajectory.flags["budget_cut_call"] = bool(
            budget_state.flags.get("budget_exceeded_call") or budget_state.flags.get("budget_blocked_model_call")
        )
        trajectory.flags["budget_cut_step"] = bool(budget_state.flags.get("budget_exceeded_step"))
        trajectory.flags["budget_cut_any"] = bool(
            trajectory.flags["budget_cut_tok"] or trajectory.flags["budget_cut_call"] or trajectory.flags["budget_cut_step"]
        )

        if close_env:
            self.env.close()
            self.env = None
        return trajectory


class GreedyAgent(LLMBasedAgent):
    """Greedy baseline: deterministic LLM decoding (temperature=0)."""

    def _temperature(self) -> float:
        return 0.0


class SamplingAgent(LLMBasedAgent):
    """Sampling baseline: stochastic LLM decoding (temperature>0)."""

    def _temperature(self) -> float:
        return max(0.0, self.config.temperature)


class RMBestOfNAgent(BaselineAgent):
    """Best-of-N sampling with RM scoring."""

    def __init__(
        self,
        config: BaselineConfig,
        env_config: Dict[str, Any],
        rm_checkpoint: str,
    ):
        super().__init__(config, env_config)
        self.rm_checkpoint = rm_checkpoint
        self.num_samples = config.num_samples
        self.rm_scorer = RMScorer(rm_checkpoint=rm_checkpoint)

    def run_episode(
        self,
        task_id: Optional[str] = None,
        query: Optional[str] = None,
        split_id: str = "eval",
        episode_seed: Optional[int] = None,
        budget: Optional[BudgetState] = None,
        prefix_actions: Optional[List[Action]] = None,
        max_policy_steps: Optional[int] = None,
        close_env: bool = False,
    ) -> Trajectory:
        base_seed = self.config.seed if episode_seed is None else episode_seed
        base_budget_snapshot = budget.snapshot() if budget is not None else None

        trajectories: List[Trajectory] = []
        for i in range(self.num_samples):
            sample_seed = self.derive_child_seed(base_seed, i + 1)
            sample_budget = budget.clone() if budget is not None else None
            agent = SamplingAgent(self.config, self.env_config)
            traj = agent.run_episode(
                task_id=task_id,
                query=query,
                split_id=split_id,
                episode_seed=sample_seed,
                budget=sample_budget,
                prefix_actions=prefix_actions,
                max_policy_steps=max_policy_steps,
                close_env=True,
            )
            trajectories.append(traj)

        picked = self._pick_best_by_rm(trajectories)
        if budget is not None and base_budget_snapshot is not None and picked.cost is not None:
            budget.tok += max(0, int(picked.cost.get("tok", 0)) - int(base_budget_snapshot["tok"]))
            budget.call += max(0, int(picked.cost.get("call", 0)) - int(base_budget_snapshot["call"]))
            budget.step += max(0, int(picked.cost.get("step", 0)) - int(base_budget_snapshot["step"]))
            if budget.exceeded():
                budget.flags["budget_exceeded"] = True

        return picked

    def _pick_best_by_rm(self, trajectories: List[Trajectory]) -> Trajectory:
        best_i = 0
        best_score = float("-inf")
        for i, traj in enumerate(trajectories):
            rm = self.rm_scorer.score(traj)
            cost = float((traj.cost or {}).get("tok", 0.0))
            bon_score = rm - 0.0 * cost
            traj.flags = traj.flags or {}
            traj.flags["rm_score"] = rm
            traj.flags["bon_score"] = bon_score
            traj.flags["bon_index"] = i
            if bon_score > best_score:
                best_score = bon_score
                best_i = i

        picked = trajectories[best_i]
        picked.flags = picked.flags or {}
        picked.flags["bon_candidates"] = len(trajectories)
        picked.flags["picked_index"] = best_i
        return picked


class RMMCTSAgent(BaselineAgent):
    """Open-loop MCTS baseline with RM scoring."""

    def __init__(
        self,
        config: BaselineConfig,
        env_config: Dict[str, Any],
        rm_checkpoint: str,
        num_simulations: Optional[int] = None,
    ):
        super().__init__(config, env_config)
        self.rm_checkpoint = rm_checkpoint
        self.num_simulations = num_simulations or config.mcts_num_simulations
        self.branching = config.mcts_branching
        self.exploration = config.mcts_exploration
        self.rm_scorer = RMScorer(rm_checkpoint=rm_checkpoint)

    def _uct(self, parent: MCTSNode, child: MCTSNode) -> float:
        if child.visits <= 0:
            return float("inf")
        exploit = child.mean_value
        explore = self.exploration * math.sqrt(math.log(parent.visits + 1) / child.visits)
        return exploit + explore

    def _select_path(self, root: MCTSNode) -> List[MCTSNode]:
        path = [root]
        node = root
        while node.children:
            node = max(node.children.values(), key=lambda child: self._uct(path[-1], child))
            path.append(node)
            if node.visits == 0:
                break
        return path

    def _propose_children(
        self,
        node: MCTSNode,
        task_id: Optional[str],
        query: Optional[str],
        split_id: str,
        base_seed: int,
        planning_budget: BudgetState,
    ) -> None:
        if node.children:
            return

        for i in range(self.branching):
            if planning_budget.exceeded() or planning_budget.remaining_steps() <= 0:
                planning_budget.flags["mcts_planning_stopped_on_budget"] = True
                break
            proposal_seed = self.derive_child_seed(base_seed, i + 1 + len(node.prefix) * 17)
            proposer = SamplingAgent(self.config, self.env_config)
            proposal_traj = proposer.run_episode(
                task_id=task_id,
                query=query,
                split_id=split_id,
                episode_seed=proposal_seed,
                budget=planning_budget,
                prefix_actions=node.prefix,
                max_policy_steps=1,
                close_env=True,
            )
            action_str = None
            for trace in reversed(proposal_traj.env_trace):
                if not trace.get("replay", False):
                    action_str = str(trace.get("action", ""))
                    break
            action = Action.from_string(action_str) if action_str else Action(action_type="finish", args={})
            key = action.to_string()
            if key not in node.children:
                node.children[key] = MCTSNode(prefix=node.prefix + [action])

    def _simulate_rollout(
        self,
        prefix: List[Action],
        task_id: Optional[str],
        query: Optional[str],
        split_id: str,
        seed: int,
        planning_budget: BudgetState,
    ) -> Trajectory:
        sampler = SamplingAgent(self.config, self.env_config)
        return sampler.run_episode(
            task_id=task_id,
            query=query,
            split_id=split_id,
            episode_seed=seed,
            budget=planning_budget,
            prefix_actions=prefix,
            close_env=True,
        )

    @staticmethod
    def _budget_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, float]:
        return {
            "call": float(max(0, int(after.get("call", 0)) - int(before.get("call", 0)))),
            "tok": float(max(0, int(after.get("tok", 0)) - int(before.get("tok", 0)))),
            "step": float(max(0, int(after.get("step", 0)) - int(before.get("step", 0)))),
        }

    def run_episode(
        self,
        task_id: Optional[str] = None,
        query: Optional[str] = None,
        split_id: str = "eval",
        episode_seed: Optional[int] = None,
        budget: Optional[BudgetState] = None,
        prefix_actions: Optional[List[Action]] = None,
        max_policy_steps: Optional[int] = None,
        close_env: bool = False,
    ) -> Trajectory:
        del max_policy_steps, close_env

        shared_budget = budget if budget is not None else self._new_budget_state()
        root_prefix = list(prefix_actions or [])
        root = MCTSNode(prefix=root_prefix)
        base_seed = self.config.seed if episode_seed is None else episode_seed

        planning_calls = 0.0
        planning_tokens = 0.0
        planning_steps = 0.0

        simulations = max(1, int(self.num_simulations))
        for sim_idx in range(simulations):
            if shared_budget.exceeded() or shared_budget.remaining_steps() <= 0:
                shared_budget.flags["mcts_planning_stopped_on_budget"] = True
                break

            path = self._select_path(root)
            leaf = path[-1]
            budget_before_propose = shared_budget.snapshot()
            self._propose_children(
                node=leaf,
                task_id=task_id,
                query=query,
                split_id=split_id,
                base_seed=self.derive_child_seed(base_seed, sim_idx + 1),
                planning_budget=shared_budget,
            )
            budget_after_propose = shared_budget.snapshot()
            delta_propose = self._budget_delta(budget_before_propose, budget_after_propose)
            planning_calls += delta_propose["call"]
            planning_tokens += delta_propose["tok"]
            planning_steps += delta_propose["step"]
            if shared_budget.exceeded() or shared_budget.remaining_steps() <= 0:
                shared_budget.flags["mcts_planning_stopped_on_budget"] = True
                break

            eval_node = leaf
            if leaf.children:
                eval_node = max(leaf.children.values(), key=lambda n: n.visits)

            rollout_seed = self.derive_child_seed(base_seed, 100000 + sim_idx)
            budget_before_rollout = shared_budget.snapshot()
            traj = self._simulate_rollout(
                prefix=eval_node.prefix,
                task_id=task_id,
                query=query,
                split_id=split_id,
                seed=rollout_seed,
                planning_budget=shared_budget,
            )
            budget_after_rollout = shared_budget.snapshot()
            delta_rollout = self._budget_delta(budget_before_rollout, budget_after_rollout)
            planning_calls += delta_rollout["call"]
            planning_tokens += delta_rollout["tok"]
            planning_steps += delta_rollout["step"]
            score = self.rm_scorer.score(traj)

            for node in path:
                node.visits += 1
                node.value_sum += score
            eval_node.visits += 1
            eval_node.value_sum += score

        if root.children:
            best_child = max(root.children.values(), key=lambda n: n.visits)
            best_prefix = best_child.prefix
        else:
            best_prefix = root_prefix

        exec_seed = self.derive_child_seed(base_seed, 999999)
        exec_agent = SamplingAgent(self.config, self.env_config)
        result = exec_agent.run_episode(
            task_id=task_id,
            query=query,
            split_id=split_id,
            episode_seed=exec_seed,
            budget=shared_budget,
            prefix_actions=best_prefix,
            close_env=True,
        )

        result.flags = result.flags or {}
        result.flags["mcts_num_simulations"] = simulations
        result.flags["mcts_branching"] = self.branching
        result.flags["mcts_root_children"] = len(root.children)
        result.flags["mcts_selected_prefix"] = [a.to_string() for a in best_prefix]
        result.flags["mcts_planning_calls"] = planning_calls
        result.flags["mcts_planning_tokens"] = planning_tokens
        result.flags["mcts_planning_steps"] = planning_steps
        if shared_budget.exceeded():
            result.flags["budget_exceeded"] = True
        return result


def _resolve_runtime_env_config(env_config: Dict[str, Any], max_steps: int) -> Dict[str, Any]:
    runtime: Dict[str, Any] = {}
    if isinstance(env_config.get("runtime"), dict):
        runtime.update(env_config["runtime"])
    for key in ("max_steps", "observation_mode", "num_products"):
        if key in env_config:
            runtime[key] = env_config[key]
    task_spec = env_config.get("task_spec", {})
    if isinstance(task_spec, dict) and "max_steps" in task_spec:
        runtime.setdefault("max_steps", int(task_spec["max_steps"]))

    runtime["max_steps"] = max_steps
    runtime.setdefault("observation_mode", "text")
    runtime.setdefault("num_products", 1000)
    return runtime


def _build_agent(
    config: BaselineConfig,
    runtime_env_config: Dict[str, Any],
    rm_checkpoint: Optional[str],
) -> BaselineAgent:
    method = config.method
    if method == "greedy":
        return GreedyAgent(config, runtime_env_config)
    if method == "sampling":
        return SamplingAgent(config, runtime_env_config)
    if method == "rm_bon":
        if rm_checkpoint is None:
            raise ValueError("rm_checkpoint required for rm_bon method")
        return RMBestOfNAgent(config, runtime_env_config, rm_checkpoint)
    if method == "rm_mcts":
        if rm_checkpoint is None:
            raise ValueError("rm_checkpoint required for rm_mcts method")
        return RMMCTSAgent(config, runtime_env_config, rm_checkpoint, num_simulations=config.mcts_num_simulations)
    raise ValueError(f"Unknown method: {method}")


def run_baseline_experiment(
    method: str,
    env_config: Dict[str, Any],
    task_manifest: Dict[str, Any],
    split: str = "S_final_test",
    seed: int = 42,
    num_samples: int = 1,
    rm_checkpoint: Optional[str] = None,
    output_dir: str = "results/exp_logs",
    temperature: float = 0.7,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4.1-mini",
    llm_top_p: Optional[float] = None,
    llm_stop: Optional[str] = None,
    llm_reasoning_effort: Optional[str] = None,
    llm_tools_enabled: bool = False,
    max_steps: int = 80,
    max_tokens: int = 6000,
    max_calls: int = 12,
    max_tasks: Optional[int] = None,
) -> Dict[str, Any]:
    random.seed(seed)
    config = BaselineConfig(
        method=method,
        seed=seed,
        num_samples=num_samples,
        temperature=temperature,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_top_p=llm_top_p,
        llm_stop=llm_stop,
        llm_reasoning_effort=llm_reasoning_effort,
        llm_tools_enabled=llm_tools_enabled,
        max_steps=max_steps,
        max_tokens=max_tokens,
        max_calls=max_calls,
    )
    runtime_env_config = _resolve_runtime_env_config(env_config=env_config, max_steps=config.max_steps)
    agent = _build_agent(config=config, runtime_env_config=runtime_env_config, rm_checkpoint=rm_checkpoint)

    task_ids = list(task_manifest["splits"][split]["task_ids"])
    if max_tasks is not None and max_tasks > 0:
        task_ids = task_ids[:max_tasks]
    trajectories: List[Trajectory] = []
    for task_id in task_ids:
        task_seed = BaselineAgent.derive_task_seed(config.seed, str(task_id))
        trajectories.append(
            agent.run_episode(
                task_id=task_id,
                split_id=split,
                episode_seed=task_seed,
            )
        )

    success_rate = sum(bool(t.env_success) for t in trajectories) / max(1, len(trajectories))
    avg_steps = sum((t.cost or {}).get("step", 0.0) for t in trajectories) / max(1, len(trajectories))
    avg_env_steps = avg_steps
    avg_policy_steps = sum(len(t.intermediate_outputs or []) for t in trajectories) / max(1, len(trajectories))
    avg_calls = sum((t.cost or {}).get("call", 0.0) for t in trajectories) / max(1, len(trajectories))
    avg_tokens = sum((t.cost or {}).get("tok", 0.0) for t in trajectories) / max(1, len(trajectories))
    avg_prompt_tokens = (
        sum(float((t.flags or {}).get("model_prompt_tokens", 0.0)) for t in trajectories) / max(1, len(trajectories))
    )
    avg_completion_tokens = (
        sum(float((t.flags or {}).get("model_completion_tokens", 0.0)) for t in trajectories)
        / max(1, len(trajectories))
    )
    avg_latency_ms = (
        sum(float((t.flags or {}).get("model_latency_ms_total", 0.0)) for t in trajectories) / max(1, len(trajectories))
    )
    total_model_calls = sum(int((t.flags or {}).get("model_call_count", 0)) for t in trajectories)
    avg_latency_per_call_ms = (
        sum(float((t.flags or {}).get("model_latency_ms_total", 0.0)) for t in trajectories) / total_model_calls
        if total_model_calls > 0
        else 0.0
    )
    repair_trigger_rate = (
        sum(int((t.flags or {}).get("repair_attempts", 0) > 0) for t in trajectories) / max(1, len(trajectories))
    )
    repair_success_rate = (
        sum(int((t.flags or {}).get("repair_successes", 0) > 0) for t in trajectories) / max(1, len(trajectories))
    )
    fallback_rate = (
        sum(int((t.flags or {}).get("fallback_count", 0) > 0) for t in trajectories) / max(1, len(trajectories))
    )
    budget_cut_rate = (
        sum(int(bool((t.flags or {}).get("budget_cut_any", False))) for t in trajectories)
        / max(1, len(trajectories))
    )
    parse_fail_rate = (
        sum(bool((t.flags or {}).get("parse_fail", False)) for t in trajectories) / max(1, len(trajectories))
    )

    exp_dir = Path(output_dir)
    if exp_dir.name == "exp_logs":
        exp_dir = exp_dir / f"{method}_{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    create_run_manifest(
        str(exp_dir),
        experiment_id=f"{method}_{split}_{seed}",
        config={"baseline": asdict(config), "env": runtime_env_config},
        method=method,
        split=split,
        num_tasks=len(task_ids),
    )

    traj_file = exp_dir / "trajectories.jsonl"
    if traj_file.exists():
        traj_file.unlink()
    for traj in trajectories:
        log_trajectory(traj, str(traj_file))

    metrics = {
        "success_rate": success_rate,
        "parse_fail_rate": parse_fail_rate,
        "num_trajectories": len(trajectories),
        "avg_step_count": avg_steps,
        "avg_env_step": avg_env_steps,
        "avg_policy_step": avg_policy_steps,
        "avg_llm_calls": avg_calls,
        "avg_tokens": avg_tokens,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "avg_latency_ms_total": avg_latency_ms,
        "avg_latency_ms_per_call": avg_latency_per_call_ms,
        "repair_trigger_rate": repair_trigger_rate,
        "repair_success_rate": repair_success_rate,
        "fallback_rate": fallback_rate,
        "budget_cut_rate": budget_cut_rate,
        "method": method,
        "split": split,
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "method": method,
        "split": split,
        "num_tasks": len(task_ids),
        "success_rate": success_rate,
        "trajectories": trajectories,
        "metrics": metrics,
    }


def run_baseline_methods(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Backward-compatible alias for earlier API names."""
    return run_baseline_experiment(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--method", type=str, required=True, help="Baseline method")
    parser.add_argument("--env_config", type=str, default="configs/env/webshop_v0.1.0.json")
    parser.add_argument("--task_manifest", type=str, default="configs/env/webshop_task_manifest_v0.1.0.json")
    parser.add_argument("--split", type=str, default="S_final_test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--llm_provider", type=str, default="openai")
    parser.add_argument("--llm_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--llm_top_p", type=float, default=None)
    parser.add_argument("--llm_stop", type=str, default=None)
    parser.add_argument("--llm_reasoning_effort", type=str, default=None)
    parser.add_argument("--llm_tools_enabled", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--max_calls", type=int, default=None)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--rm_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/exp_logs")
    args = parser.parse_args()

    env_config = load_config(args.env_config)
    task_manifest = load_config(args.task_manifest)
    budget_cfg = env_config.get("budget_limits", {}) if isinstance(env_config, dict) else {}
    task_cfg = env_config.get("task_spec", {}) if isinstance(env_config, dict) else {}

    max_steps = args.max_steps if args.max_steps is not None else int(task_cfg.get("max_steps", 80))
    max_tokens = args.max_tokens if args.max_tokens is not None else int(budget_cfg.get("B_tok", 6000))
    max_calls = args.max_calls if args.max_calls is not None else int(budget_cfg.get("B_call", 12))

    results = run_baseline_experiment(
        method=args.method,
        env_config=env_config,
        task_manifest=task_manifest,
        split=args.split,
        seed=args.seed,
        num_samples=args.num_samples,
        rm_checkpoint=args.rm_checkpoint,
        output_dir=args.output_dir,
        temperature=args.temperature,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_top_p=args.llm_top_p,
        llm_stop=args.llm_stop,
        llm_reasoning_effort=args.llm_reasoning_effort,
        llm_tools_enabled=bool(args.llm_tools_enabled),
        max_steps=max_steps,
        max_tokens=max_tokens,
        max_calls=max_calls,
        max_tasks=args.max_tasks,
    )
    print(f"Success rate: {results['success_rate']:.2%}")


if __name__ == "__main__":
    main()
