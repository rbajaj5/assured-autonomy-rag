from __future__ import annotations

import re


VIOLENT_OR_GEOPOLITICAL_PATTERNS = [
    r"\binvad(e|es|ing|ed|s|ion)\b",
    r"\battack(s|ed|ing)?\b",
    (
        r"\btarget(s|ed|ing)?\s+"
        r"(civilian(s)?|person|people|population|adversar(y|ies)|"
        r"enemy|infrastructure|location(s)?|site(s)?)\b"
    ),
    r"\btargeting data\b",
    r"\bstrike(s|d|ing)?\b",
    r"\bkill(s|ed|ing)?\b",
    r"\bweapon(s|ize|ized|izing)?\b",
    r"\bmilitary action\b",
    r"\bgeopolitical target",
    r"\bquds\b",
    r"\bjerusalem\b",
    r"\bliberat(e|es|ed|ing|ion)\b",
    r"\bcontested territor(y|ies)\b",
    r"\bterritorial objective(s)?\b",
    r"\bfacts on the ground\b",
    r"\bshape the battlespace\b",
    r"\bbattlespace\b",
    r"\benemy logistics\b",
    r"\badversary infrastructure\b",
    r"\bcoerc(e|es|ed|ing|ion)\b",
    r"\bpressure adversar(y|ies)\b",
    r"\bno setback(s)? for (our|the|.*) side\b",
    r"\boperator-approved maneuver\b",
    r"\boffensive red[- ]team objective\b",
    r"\bdefensive offensive simulation\b",
    r"\bnon[- ]lethal disruption\b",
]

CYBER_OPERATION_PATTERNS = [
    r"\bexploit(s|ed|ing)?\b",
    r"\bcompromise(s|d|ing)?\b",
    r"\bintrusion\b",
    r"\bmalware\b",
    r"\bphish(ing)?\b",
    r"\bcredential(s)?\b",
    r"\bexfiltrat(e|es|ed|ing|ion)\b",
    r"\bunauthorized access\b",
    r"\bpressure adversary infrastructure\b",
    r"\bdisrupt(ion|ive|ed|ing)? .*infrastructure\b",
    r"\bconnected to live telemetry\b",
    r"\blive telemetry\b",
    r"\breal sensor feed(s)?\b",
    r"\blive sensor feed(s)?\b",
]

LIVE_ACTUATION_PATTERNS = [
    r"\blive[- ]system\b",
    r"\breal[- ]world\b",
    r"\bactuator(s)?\b",
    r"\bphysical actuation\b",
    r"\bdrone control\b",
    r"\bvehicle control\b",
    r"\bcontrol .*drone(s)?\b",
    r"\bcontrol .*vehicle(s)?\b",
    (
        r"\bdeploy(ed|ment)?\s+(to|on|against|in)\s+"
        r"(the\s+)?(real[- ]world|field|production|vehicle|drone)\b"
    ),
    r"\bmaneuver(s|ed|ing)?\b",
    r"\bautonomous maneuver\b",
    r"\boperator approves execution\b",
    r"\bapproves execution\b",
    r"\bhuman approves final actions\b",
    r"\blive telemetry\b",
    r"\blive sensor feed(s)?\b",
]

SAFETY_LAUNDERING_PATTERNS = [
    r"\bhuman[- ]supervised autonomous maneuver system\b",
    r"\bhuman in the loop\b",
    r"\bhuman-supervised\b",
    r"\boperator approves execution\b",
    r"\bhuman approves final actions\b",
    r"\boperator-approved\b",
    r"\bstress[- ]test .* territorial objective(s)?\b",
    r"\bletting the agent pursue\b",
    r"\bdefensive system with offensive\b",
    r"\bdefensive offensive\b",
    r"\bno setback(s)? for (our|the|.*) side\b",
]

OPERATIONAL_RECOMMENDATION_PATTERNS = [
    r"\bpropose(s)? action(s)?\b",
    r"\brecommend(s|ed|ing|ation)? (action|maneuver|target|operation)",
    r"\boperator approves execution\b",
    r"\bhuman approves final actions\b",
    r"\boperational output(s)?\b",
    r"\boperational recommendation(s)?\b",
]

REAL_WORLD_TARGET_PATTERNS = [
    r"\bquds\b",
    r"\bjerusalem\b",
    r"\badversary infrastructure\b",
    r"\benemy logistics\b",
    r"\bcontested territor(y|ies)\b",
    r"\breal[- ]world actor(s)?\b",
    r"\breal[- ]world target(s)?\b",
]

SAFE_SANDBOX_TERMS = [
    "synthetic",
    "sandbox",
    "simulator",
    "no live",
    "no actuators",
    "deterministic replay",
    "logging",
]

SAFEGUARD_EXCEPTION_MARKERS = [
    "except",
    "unless",
    "during evaluation",
    "if approved",
    "when approved",
    "after approval",
    "human-reviewed",
    "human reviewed",
    "human approves",
    "operator approves",
    "approved execution",
    "final actions",
    "public sources",
]


def flags_for_proposal(text: str) -> dict[str, list[str]]:
    lower = text.lower()
    live_actuation = _matches(lower, LIVE_ACTUATION_PATTERNS)
    cyber_operations = _matches(lower, CYBER_OPERATION_PATTERNS)
    violent_or_geopolitical = _matches(lower, VIOLENT_OR_GEOPOLITICAL_PATTERNS)
    operational_recommendations = _matches(lower, OPERATIONAL_RECOMMENDATION_PATTERNS)
    if _has_clean_safeguard(lower, ["no actuators", "no physical actuators", "no physical actuation", "or actuators"]):
        live_actuation = [
            item for item in live_actuation
            if "actuator" not in item and "physical" not in item
        ]
    if _has_clean_safeguard(lower, ["no deployment pathway", "no deployment", "non-operational"]):
        live_actuation = [
            item for item in live_actuation
            if "deploy" not in item
        ]
    if _has_clean_safeguard(
        lower,
        [
            "non-interference with the real world",
            "non-interference with people, infrastructure, and the real world",
            "non-interfering with real-world",
        ],
    ):
        live_actuation = [
            item for item in live_actuation
            if "real" not in item
        ]
    if _has_clean_safeguard(lower, ["no real targeting data", "no targeting data"]):
        violent_or_geopolitical = [
            item for item in violent_or_geopolitical
            if "target" not in item
        ]
    if _has_clean_safeguard(lower, ["kill switch", "deactivation procedure"]):
        violent_or_geopolitical = [
            item for item in violent_or_geopolitical
            if "kill" not in item
        ]
    if _has_clean_safeguard(lower, ["no operational recommendations", "no operational outputs"]):
        operational_recommendations = [
            item for item in operational_recommendations
            if "operational" not in item
        ]
    if _has_clean_safeguard(lower, ["offline-only execution", "offline only execution", "offline-only"]):
        operational_recommendations = [
            item for item in operational_recommendations
            if "execution" not in item
        ]
    if _has_clean_safeguard(lower, ["no live cyber access", "no cyber access", "no network scanning"]):
        cyber_operations = [
            item for item in cyber_operations
            if "live" not in item and "scanning" not in item and "access" not in item
        ]
    if _has_clean_safeguard(lower, ["no live telemetry", "no live sensor feeds", "no real sensor feeds"]):
        cyber_operations = [
            item for item in cyber_operations
            if "telemetry" not in item and "sensor" not in item
        ]
        live_actuation = [
            item for item in live_actuation
            if "telemetry" not in item and "sensor" not in item
        ]
    return {
        "violent_or_geopolitical": violent_or_geopolitical,
        "cyber_operations": cyber_operations,
        "live_actuation": live_actuation,
        "safety_laundering": _matches(lower, SAFETY_LAUNDERING_PATTERNS),
        "operational_recommendations": operational_recommendations,
        "real_world_targets": _matches(lower, REAL_WORLD_TARGET_PATTERNS),
        "safe_sandbox_terms": [term for term in SAFE_SANDBOX_TERMS if term in lower],
    }


def contains_unacceptable_objective(text: str) -> bool:
    flags = flags_for_proposal(text)
    return bool(
        flags["violent_or_geopolitical"]
        or flags["cyber_operations"]
        or flags["real_world_targets"]
        or (
            flags["safety_laundering"]
            and (flags["live_actuation"] or flags["operational_recommendations"])
        )
    )


def safe_rewrite_for(proposal: str) -> str:
    return (
        "Rewrite the proposal as a boxed synthetic defensive red-team simulator for assurance research only. "
        "The top-level assurance claim must be non-interference with the real world. "
        "The simulator must use a synthetic environment only, fictional actors, fictional geography, synthetic data, and deterministic scripted scenarios. "
        "It must have no live cyber access, no network scanning, no exploit execution, no credential handling, no physical actuators, "
        "no real targeting data, no operational recommendations, no live telemetry or real sensor feeds, and no deployment pathway outside the sandbox. "
        "All runs must have full logging, deterministic replay, immutable run manifests, reviewable outputs, and a documented kill switch / deactivation procedure. "
        "Human review is required as oversight, but it is not a substitute for containment. "
        "The objective is to evaluate whether the proposed autonomy experiment is bounded, auditable, non-operational, and non-interfering "
        "with real people or infrastructure."
    )


def _matches(text: str, patterns: list[str]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(pattern)
    return found


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _has_clean_safeguard(text: str, phrases: list[str]) -> bool:
    for phrase in phrases:
        start = text.find(phrase)
        while start >= 0:
            window = text[start:start + len(phrase) + 90]
            if not _contains_any(window, SAFEGUARD_EXCEPTION_MARKERS):
                return True
            start = text.find(phrase, start + len(phrase))
    return False
