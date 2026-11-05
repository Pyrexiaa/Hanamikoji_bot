"""Shared handcrafted baseline agents for both Tiny and Medium Hanamikoji."""

from .lead_protector_agent import LeadProtectorAgent
from .comeback_contester_agent import ComebackContesterAgent
from .rank_weighted_agent import RankWeightedAgent
from .majority_threshold_agent import MajorityThresholdAgent
from .denial_first_agent import DenialFirstAgent
from .action_order_template_agent import ActionOrderTemplateAgent
from .pair_synergy_agent import PairSynergyAgent
from .risk_adaptive_agent import RiskAdaptiveAgent
from .belief_bucket_agent import BeliefBucketAgent
from .coverage_first_agent import CoverageFirstAgent


__all__ = [
    "LeadProtectorAgent",
    "ComebackContesterAgent",
    "RankWeightedAgent",
    "MajorityThresholdAgent",
    "DenialFirstAgent",
    "ActionOrderTemplateAgent",
    "PairSynergyAgent",
    "RiskAdaptiveAgent",
    "BeliefBucketAgent",
    "CoverageFirstAgent",
]
