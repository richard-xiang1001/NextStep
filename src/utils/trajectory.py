"""Trajectory logging and summarization utilities."""

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TrajectorySummary:
    """
    Summarized trajectory for reward model input.

    Follows the TrajectorySummarySchema from research_spec.md v0.1.0
    """

    query_text: str
    workflow_steps: List[Dict[str, Any]]
    step_outputs: List[Any]  # Truncated worker/model outputs
    env_trace_summary: List[Dict[str, str]]  # (obs, action) tuples
    terminal_output: str
    budget_flags: Dict[str, bool]

    # Schema constraints (v0.1.0)
    MAX_CHARS_PER_STEP = 600
    MAX_STEPS = 6
    MAX_ENV_TRACE_ENTRIES = 80
    MAX_CHARS_OBS_ACTION = 240

    @staticmethod
    def _to_text(value: Any) -> str:
        """Serialize arbitrary output value into a deterministic string."""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(value)

    def truncate_output(self, text: Any, max_chars: int) -> str:
        """Truncate text to max characters."""
        text = self._to_text(text)
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with truncation applied."""
        return {
            "query_text": self.query_text,
            "workflow_steps": self.workflow_steps[: self.MAX_STEPS],
            "step_outputs": [
                self.truncate_output(out, self.MAX_CHARS_PER_STEP) for out in self.step_outputs[: self.MAX_STEPS]
            ],
            "env_trace_summary": [
                {
                    "obs": self.truncate_content(trace["obs"], self.MAX_CHARS_OBS_ACTION),
                    "action": self.truncate_content(trace["action"], self.MAX_CHARS_OBS_ACTION),
                }
                for trace in self.env_trace_summary[: self.MAX_ENV_TRACE_ENTRIES]
            ],
            "terminal_output": self.terminal_output,
            "budget_flags": self.budget_flags,
        }

    def truncate_content(self, text: str, max_chars: int) -> str:
        """Truncate content preserving structure."""
        if not text or len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def to_json(self) -> str:
        """Convert to JSON string with deterministic key ordering."""
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass
class Trajectory:
    """Complete trajectory logging for debugging and analysis."""

    qid: str
    query_text: str
    split_id: str  # S_rm_train, S_policy_train, etc.
    workflow: List[Dict[str, Any]]
    intermediate_outputs: List[Any]
    env_trace: List[Dict[str, Any]]
    rm_scores: Optional[Dict[str, float]] = None
    cost: Optional[Dict[str, float]] = None
    env_success: Optional[bool] = None
    debug_only: bool = True  # Mark as debug when used for training
    flags: Optional[Dict[str, bool]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL logging."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    def to_summary(self) -> TrajectorySummary:
        """Convert to TrajectorySummary for RM input."""
        terminal_output = ""
        if self.intermediate_outputs:
            terminal_output = TrajectorySummary._to_text(self.intermediate_outputs[-1])
        budget_exceeded = False
        if self.flags:
            budget_exceeded = bool(
                self.flags.get("budget_truncated", False)
                or self.flags.get("budget_exceeded", False)
                or self.flags.get("budget_exceeded_tok", False)
                or self.flags.get("budget_exceeded_call", False)
                or self.flags.get("budget_exceeded_step", False)
                or self.flags.get("budget_blocked_model_call", False)
                or self.flags.get("budget_blocked_model_token", False)
            )

        return TrajectorySummary(
            query_text=self.query_text,
            workflow_steps=self.workflow,
            step_outputs=self.intermediate_outputs,
            env_trace_summary=[
                {"obs": step.get("obs", ""), "action": step.get("action", "")} for step in self.env_trace
            ],
            terminal_output=terminal_output,
            budget_flags={
                "parse_fail": self.flags.get("parse_fail", False) if self.flags else False,
                "timeout": self.flags.get("timeout", False) if self.flags else False,
                "budget_truncated": budget_exceeded,
            },
        )


def log_trajectory(trajectory: Trajectory, output_file: str):
    """
    Append trajectory to JSONL log file.

    Args:
        trajectory: Trajectory to log
        output_file: Path to output JSONL file
    """
    with open(output_file, "a") as f:
        f.write(trajectory.to_json() + "\n")


def load_trajectories(input_file: str) -> List[Trajectory]:
    """
    Load trajectories from JSONL file.

    Args:
        input_file: Path to input JSONL file

    Returns:
        List of Trajectory objects
    """
    trajectories = []
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            trajectories.append(Trajectory(**data))
    return trajectories
