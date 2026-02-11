"""Base environment interface for interactive tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Action:
    """Represents an action in the environment."""

    action_type: str  # e.g., "search", "click", "finish"
    args: Dict[str, Any]

    def to_string(self) -> str:
        """Convert action to string representation for logging."""
        if self.action_type == "search":
            return f"search[{self.args.get('query', '')}]"
        elif self.action_type == "click":
            return f"click[{self.args.get('element_key', '')}]"
        elif self.action_type == "finish":
            return f"finish"
        else:
            return f"{self.action_type}[{self.args}]"

    @classmethod
    def from_string(cls, action_str: str) -> "Action":
        """Parse action from string representation."""
        action_str = action_str.strip()
        if action_str.startswith("search["):
            query = action_str[7:-1]
            return cls(action_type="search", args={"query": query})
        elif action_str.startswith("click["):
            element = action_str[6:-1]
            return cls(action_type="click", args={"element_key": element})
        elif action_str == "finish":
            return cls(action_type="finish", args={})
        else:
            # Fallback for other action types
            return cls(action_type="unknown", args={"raw": action_str})


@dataclass
class Observation:
    """Represents an observation from the environment."""

    text: str  # Text observation
    html: Optional[str] = None  # HTML observation (if available)
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata

    def __str__(self) -> str:
        return self.text


@dataclass
class StepResult:
    """Result of a single environment step."""

    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class BaseEnvironment(ABC):
    """
    Base interface for interactive environments.

    This provides a unified interface for different interactive benchmarks
    like WebShop, ALFWorld, ScienceWorld, etc.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the environment.

        Args:
            config: Environment configuration dict
        """
        self.config = config
        self.max_steps = config.get("max_steps", 80)
        self.current_step = 0

    @abstractmethod
    def reset(self, task_id: Optional[str] = None) -> Tuple[Observation, Dict[str, Any]]:
        """
        Reset the environment for a new task.

        Args:
            task_id: Optional specific task ID to reset to

        Returns:
            (initial_observation, info)
        """
        pass

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """
        Execute one action in the environment.

        Args:
            action: Action to execute

        Returns:
            StepResult containing observation, reward, done, info
        """
        pass

    @abstractmethod
    def get_task_query(self) -> str:
        """
        Get the natural language query/instruction for the current task.

        Returns:
            Task query string
        """
        pass

    @abstractmethod
    def get_success(self) -> bool:
        """
        Check if the current task was completed successfully.

        Returns:
            True if task was successful, False otherwise
        """
        pass

    def get_step_count(self) -> int:
        """Get the number of steps taken in the current episode."""
        return self.current_step

    def is_budget_exceeded(
        self,
        token_count: int = 0,
        call_count: int = 0,
        max_tokens: int = 6000,
        max_calls: int = 12,
    ) -> bool:
        """
        Check if budget limits are exceeded.

        Args:
            token_count: Current token count
            call_count: Current API call count
            max_tokens: Maximum allowed tokens
            max_calls: Maximum allowed API calls

        Returns:
            True if any budget is exceeded
        """
        if self.current_step >= self.max_steps:
            return True
        if token_count >= max_tokens:
            return True
        if call_count >= max_calls:
            return True
        return False

    def get_available_actions(self) -> List[str]:
        """
        Get list of available actions at current state.

        Returns:
            List of action descriptions
        """
        # Default implementation - override in specific environments
        return ["search[query]", "click[element]", "finish"]

    def close(self):
        """Clean up environment resources."""
        pass


class EnvironmentError(Exception):
    """Base exception for environment errors."""

    pass


class BudgetExceededError(EnvironmentError):
    """Raised when budget limits are exceeded."""

    pass


class ParseError(EnvironmentError):
    """Raised when action parsing fails."""

    pass
