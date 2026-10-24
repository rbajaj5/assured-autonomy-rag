"""Syllabus navigation for EN.695.715 Assured Autonomy modules."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import AA_MODULES

MODULE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "M01-02": ("assured autonomy", "ai safety", "concrete problems", "policy primer"),
    "M03": ("security", "privacy", "cybersecurity", "attack", "defense", "data leakage", "access control"),
    "M04": ("anomaly", "fault", "failure detection", "time series", "incident", "abnormal", "outlier"),
    "M05": ("dataset shift", "distribution shift", "ood", "out-of-distribution", "drift", "covariate shift"),
    "M06": ("formal verification", "verification", "validation", "reachability", "reluplex", "verisig", "claim checking"),
    "M07": ("test", "evaluation", "simulation", "robustness certification", "deepxplore", "benchmark", "scenario"),
    "M08": ("interpretability", "explainable", "explanation", "lime", "transparent", "audit report", "report"),
    "M09": ("human-ai", "human in the loop", "coordination", "trust", "oversight", "human review"),
    "M10": ("policy", "governance", "accountability", "responsible ai", "transparency", "audit", "law", "police"),
    "M11": ("adversarial", "evasion", "black-box", "transferability", "safety chip", "compromised", "hijack"),
    "M12": ("runtime assurance", "runtime monitoring", "monitor", "simplex", "soter", "runtime verification", "fallback"),
    "M13": ("safe reinforcement learning", "safe rl", "constraint", "action projection"),
    "M14": ("multi-agent", "cooperative", "formation control", "marl", "swarm", "fleet", "agent"),
}


@dataclass(frozen=True)
class ModuleMatch:
    module_id: str
    module_name: str
    score: float
    evidence: list[str]


class SyllabusNavigator:
    """Maps free-text queries to module numbers from the course reading list."""

    def match(self, query: str, *, top_k: int = 3) -> list[ModuleMatch]:
        text = query.lower()
        matches: list[ModuleMatch] = []
        for module_id, keywords in MODULE_KEYWORDS.items():
            evidence = [kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text)]
            score = float(len(evidence))
            if module_id.lower() in text:
                score += 2.0
                evidence.append(module_id)
            if AA_MODULES[module_id].lower() in text:
                score += 2.0
                evidence.append(AA_MODULES[module_id])
            if score > 0:
                matches.append(ModuleMatch(module_id, AA_MODULES[module_id], score, evidence))
        matches.sort(key=lambda m: (-m.score, m.module_id))
        return matches[:top_k]

    def module_filter(self, query: str, *, top_k: int = 3) -> list[str] | None:
        matches = self.match(query, top_k=top_k)
        return [m.module_id for m in matches] or None
