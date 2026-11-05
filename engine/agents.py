"""Agent interfaces and baseline agent implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from .info_state import RoundInfoState
from .models import Action


class Agent(ABC):
    """Base class for AI agents using mixed strategies."""

    @abstractmethod
    def get_action_distribution(self, info_state: RoundInfoState) -> Dict[Action, float]:
        """Return a mixed strategy as ``{action: probability, ...}``."""
        raise NotImplementedError


class RandomAgent(Agent):
    """A simple random agent that works with any variant."""

    def get_action_distribution(self, info_state: RoundInfoState) -> Dict[Action, float]:
        available = list(info_state.get_legal_actions())

        if not available:
            raise ValueError("No available actions!")

        p = 1.0 / len(available)
        return {action: p for action in available}
