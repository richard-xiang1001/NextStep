"""WebShop environment wrapper."""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Add WebShop submodule to path
WEBSHOP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "third_party",
    "WebShop",
)
if WEBSHOP_PATH not in sys.path:
    sys.path.insert(0, WEBSHOP_PATH)

_WEBSHOP_IMPORT_ERROR: Optional[BaseException] = None

try:
    from web_agent_site.envs import WebAgentTextEnv
except ImportError:
    _WEBSHOP_IMPORT_ERROR = sys.exc_info()[1]
    WebAgentTextEnv = None
    print(
        "Warning: WebShop import failed: "
        f"{_WEBSHOP_IMPORT_ERROR}. Install with: cd third_party/WebShop && bash setup.sh -d small"
    )

from .base import Action, BaseEnvironment, Observation, StepResult


class WebShopEnvironment(BaseEnvironment):
    """
    Wrapper for the WebShop environment.

    WebShop is a simulated e-commerce website with 1.18M real products
    and human instructions for grounded language agent research.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize WebShop environment.

        Args:
            config: Environment configuration with keys:
                - max_steps: Maximum environment steps (default: 80)
                - observation_mode: 'text' or 'html' (default: 'text')
                - num_products: Number of products to load (1000 or all)
        """
        super().__init__(config)

        if WebAgentTextEnv is None:
            raise ImportError(
                "WebShop not installed. Run: cd third_party/WebShop && bash setup.sh -d small. "
                f"Original import error: {_WEBSHOP_IMPORT_ERROR}"
            )

        self.observation_mode = config.get("observation_mode", "text")
        self.num_products = config.get("num_products", 1000)
        self._env: Optional["WebAgentTextEnv"] = None
        self._current_task_id: Optional[str] = None

    def _init_env(self) -> "WebAgentTextEnv":
        """Initialize the underlying WebShop environment."""
        if self._env is None:
            self._env = WebAgentTextEnv(
                observation_mode=self.observation_mode,
                num_products=self.num_products,
            )
        return self._env

    @staticmethod
    def _normalize_task_id(task_id: Any) -> int:
        """
        Normalize task IDs to the integer session index expected by WebShop.

        Supported input formats:
        - int
        - numeric string, e.g. "42"
        - manifest style id, e.g. "task_00042"
        """
        if isinstance(task_id, int):
            return task_id
        if isinstance(task_id, str):
            task_id = task_id.strip()
            if task_id.startswith("task_"):
                suffix = task_id.split("_", 1)[1]
                return int(suffix)
            return int(task_id)
        raise ValueError(f"Unsupported task_id format: {task_id!r}")

    def reset(self, task_id: Optional[str] = None) -> Tuple[Observation, Dict[str, Any]]:
        """
        Reset the environment for a new task.

        Args:
            task_id: Optional specific task ID (index in WebShop)

        Returns:
            (initial_observation, info)
        """
        env = self._init_env()

        if task_id is not None:
            # WebShop uses integer session indices.
            obs, info = env.reset(session=self._normalize_task_id(task_id))
        else:
            obs, info = env.reset()
        info = info or {}

        self.current_step = 0
        self._current_task_id = str(task_id) if task_id is not None else str(info.get("task_id", "unknown"))

        # Convert to our Observation format
        observation = Observation(text=obs, metadata={"raw_obs": obs})

        return observation, info

    def step(self, action: Action) -> StepResult:
        """
        Execute one action in WebShop.

        Args:
            action: Action to execute (search, click, etc.)

        Returns:
            StepResult with observation, reward, done, info
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Convert our Action to WebShop action string
        action_str = action.to_string()

        # Execute action
        obs, reward, done, info = self._env.step(action_str)
        info = info or {}

        self.current_step += 1

        # Convert to our format
        observation = Observation(text=obs, metadata={"raw_obs": obs})

        # Check if budget exceeded
        info["budget_exceeded"] = self.current_step >= self.max_steps

        return StepResult(
            observation=observation,
            reward=reward,
            done=done or info["budget_exceeded"],
            info=info,
        )

    def get_task_query(self) -> str:
        """
        Get the natural language instruction for the current task.

        Returns:
            Task instruction string
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # WebShop stores the instruction in the environment
        return getattr(self._env, "instruction_text", getattr(self._env, "instruction", ""))

    def get_success(self) -> bool:
        """
        Check if the current task was completed successfully.

        Returns:
            True if task was successful
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # WebShop provides success metric
        return getattr(self._env, "goal_met", False)

    def get_available_actions(self) -> List[str]:
        """
        Get available actions at current state.

        For WebShop, this depends on the current page type.
        """
        if self._env is None:
            return ["search[query]"]

        # Get available actions from WebShop
        # This is a simplified version - actual implementation would query the env
        return ["search[query]", "click[element_key]", "finish"]

    def get_action_space(self) -> Dict[str, Any]:
        """
        Get WebShop action space specification.

        Returns:
            Dict with action space info
        """
        return {
            "search": {"type": "text", "description": "Search for products"},
            "click": {"type": "discrete", "description": "Click on an element"},
            "finish": {"type": "terminal", "description": "Finish the task"},
        }

    def get_num_tasks(self) -> int:
        """Get total number of tasks in WebShop."""
        env = self._init_env()
        return getattr(env, "num_tasks", 12087)  # Default WebShop size


def create_webshop_env(config: Dict[str, Any]) -> WebShopEnvironment:
    """
    Factory function to create WebShop environment.

    Args:
        config: Configuration dict

    Returns:
        Initialized WebShopEnvironment
    """
    return WebShopEnvironment(config)
