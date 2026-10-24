from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPAIRED_DIR = Path("reports/principal_repaired_cards")
OUTPUT_DIR = Path("reports/principal_distilled_cards")
INDEX_PATH = Path("reports/principal_distilled_cards_index.md")
AUDIT_PATH = Path("reports/principal_distilled_cards_quality_audit.md")
JSON_PATH = Path("reports/principal_distilled_cards.json")


@dataclass(frozen=True)
class DistilledCardSpec:
    canonical_id: str
    filename_stem: str
    compiler_role: str
    use_when: list[str]
    do_not_use_when: list[str]
    directly_supported_claims: list[str]
    scope_conditions: list[str]
    not_supported: list[str]
    weak_reader_traps: list[str]
    backend_policy: list[str]
    evidence_packet_snippets: dict[str, str]


DISTILLED_SPECS = [
    DistilledCardSpec(
        canonical_id="M12_RuntimeAssurance-Desai-2019-SOTER.pdf",
        filename_stem="M12_RuntimeAssurance-Desai-2019-SOTER.pdf",
        compiler_role="runtime-assurance hard-stop card",
        use_when=[
            "A proposal or judge output involves live telemetry, actuation, safety envelopes, runtime monitors, or fallback logic.",
            "The compiler must decide whether an unsafe or uncertain state should block propagation.",
            "A weak model treats an uncertified high-capability component as if it can be trusted directly.",
        ],
        do_not_use_when=[
            "The question is only about broad AI safety taxonomy or legal policy framing.",
            "No monitor, decision module, fallback controller, switching authority, or safety envelope is relevant.",
            "The evidence packet would imply SOTER certifies arbitrary robots, LLMs, or deployment settings.",
        ],
        directly_supported_claims=[
            "SOTER uses a Simplex-style runtime-assurance structure with an advanced controller, safe controller, and decision module.",
            "The decision module monitors state and switches to a safe/fallback controller when safety could be violated.",
            "Runtime assurance claims are conditional on a defined safety specification, timing assumptions, and a well-formed module.",
            "The paper supports runtime enforcement around uncertified components, not blind trust in those components.",
            "Case-study evidence is bounded by modeling assumptions such as known/static obstacles in the supplied notes.",
        ],
        scope_conditions=[
            "Safety property must be explicit and monitorable.",
            "Fallback controller must be certified or justified for the modeled safe region.",
            "Decision-module timing and switching latency must match system dynamics.",
            "Sensor/state assumptions must be accurate enough for the monitor.",
            "Composition across modules requires preserving each module's assumptions.",
        ],
        not_supported=[
            "Unconditional deployment safety.",
            "Approval of live telemetry without a monitor/fallback/safety-envelope argument.",
            "Certification of the advanced controller itself.",
            "Generalization to unknown environments without additional evidence.",
            "Using human approval as a substitute for runtime containment.",
        ],
        weak_reader_traps=[
            "Equating the phrase runtime assurance with guaranteed safety.",
            "Treating fallback as useful without checking timing and switching authority.",
            "Reading a case study as a universal deployment proof.",
            "Letting high or unacceptable risk map to approve/revise because a monitor is mentioned.",
        ],
        backend_policy=[
            "Mistral: state the consistency rule before evidence: high/unacceptable risk cannot map to approve/revise.",
            "GPT-3.5: include only SOTER snippets E1-E3 when runtime hard-stop logic is needed.",
            "GPT-4.1-nano: require evidence IDs for monitor, fallback, and scope conditions.",
            "Grok: reject empty evidence_ids_used when citing SOTER as support.",
        ],
        evidence_packet_snippets={
            "E1": "SOTER supports a runtime monitor/decision-module/fallback-controller pattern for enforcing a defined safety envelope.",
            "E2": "Its assurance claim is conditional on an explicit safety property, valid timing assumptions, and a certified safe controller.",
            "E3": "SOTER does not justify approving live or unsafe operation merely because a human or monitor is mentioned.",
        },
    ),
    DistilledCardSpec(
        canonical_id="M06_ExternalReadings-2024-open_autonomy_safety_case_2024-pdf",
        filename_stem="M06_ExternalReadings-2024-open_autonomy_safety_case_2024-pdf",
        compiler_role="safety-case evidence discipline card",
        use_when=[
            "The compiler must separate an assurance claim from the evidence that supports it.",
            "A proposal uses safety-case language, assumptions, scope, or argument structure.",
            "A model appears to treat mitigation plans or labels as proof.",
        ],
        do_not_use_when=[
            "Runtime monitor, fallback-controller, or hard-stop mechanics are the primary question.",
            "The card would be used to certify deployment safety without concrete evidence.",
            "The proposal lacks a bounded system definition and operational context.",
        ],
        directly_supported_claims=[
            "A safety case links claims, arguments, evidence, assumptions, and scope.",
            "Evidence must be relevant to the claim and operating context.",
            "Assurance language is not equivalent to sufficient evidence.",
            "Scope and assumptions constrain what a safety case can support.",
            "Missing or weak evidence should block strong assurance conclusions.",
        ],
        scope_conditions=[
            "System boundary and operational context must be explicit.",
            "Claims must identify supporting evidence and assumptions.",
            "Evidence must be current, relevant, and auditable.",
            "Assurance must not exceed the evaluated context.",
            "Residual uncertainty must be visible to reviewers.",
        ],
        not_supported=[
            "Runtime hard-stop or fallback-controller design.",
            "Automatic approval because a safety case exists.",
            "Treating red-team, defensive, or stress-test labels as containment evidence.",
            "Using human oversight as proof of non-interference.",
            "Claims beyond the documented operational design domain.",
        ],
        weak_reader_traps=[
            "Confusing safety-case form with safety-case sufficiency.",
            "Treating proposed mitigations as already-verified evidence.",
            "Ignoring assumptions and operational-context limits.",
            "Using safety-case vocabulary to soften a deterministic reject.",
        ],
        backend_policy=[
            "Mistral: make unsupported-claim and evidence-sufficiency checks explicit.",
            "GPT-3.5: send only the claim/evidence/scope snippets to avoid overload.",
            "GPT-4.1-nano: distinguish caveats from unsupported factual assertions.",
            "Grok: require citation IDs for every claim/evidence statement.",
        ],
        evidence_packet_snippets={
            "E1": "A safety case supports assurance only when claims are backed by relevant argument, evidence, assumptions, and scope.",
            "E2": "The safety-case scope is limited by the stated system boundary and operational context.",
            "E3": "This card does not provide runtime-controller or fallback evidence by itself.",
        },
    ),
    DistilledCardSpec(
        canonical_id="M06_ExternalReadings-2023-waymo_safety_case_2023-pdf",
        filename_stem="M06_ExternalReadings-2023-waymo_safety_case_2023-pdf",
        compiler_role="V&V / safety-case / risk-acceptance card",
        use_when=[
            "The compiler must reason about acceptance criteria, hazards, evidence credibility, or lifecycle confidence.",
            "A proposal claims deployment readiness from limited tests.",
            "A model needs evidence-discipline support for automated-driving assurance.",
        ],
        do_not_use_when=[
            "The question requires runtime monitor/fallback-controller mechanics.",
            "The card would be read as Waymo proving deployment safety for a new system.",
            "The evidence does not match the automated-driving context or acceptance claim.",
        ],
        directly_supported_claims=[
            "Waymo-style safety cases use structured evidence, hazard reasoning, and acceptance criteria.",
            "No single metric or test is sufficient for deployment readiness.",
            "Evidence credibility and lifecycle confidence matter to the assurance argument.",
            "Acceptance depends on operational context and the Automated Driving System's intended behavior.",
            "Safety-case evidence can organize risk reasoning without replacing runtime controls.",
        ],
        scope_conditions=[
            "Hazards and acceptance criteria must be explicit.",
            "Evidence must match the operational context and system capability.",
            "Claims should be traceable to tests, analyses, or field evidence.",
            "Lifecycle monitoring can support confidence growth but is not proof by itself.",
            "Residual risk must be reviewed before acceptance.",
        ],
        not_supported=[
            "A proof that Waymo or any new system is safe to deploy.",
            "Runtime hard-stop, Simplex, or fallback-controller design.",
            "Approval from one favorable metric.",
            "Trust transfer from Waymo's domain to unrelated systems.",
            "Treating human/operator approval as containment.",
        ],
        weak_reader_traps=[
            "Mistaking a safety-case methodology for a deployment certificate.",
            "Collapsing many evidence types into one approval score.",
            "Ignoring the difference between confidence growth and final acceptance.",
            "Using V&V evidence as if it were runtime enforcement.",
        ],
        backend_policy=[
            "Mistral: preserve reject when evidence is insufficient for acceptance.",
            "GPT-3.5: include only AUR/acceptance/evidence credibility snippets.",
            "GPT-4.1-nano: do not count cautious caveats as unsupported claims.",
            "Grok: require evidence IDs for hazard and acceptance references.",
        ],
        evidence_packet_snippets={
            "E1": "Waymo-style safety cases organize hazard, evidence, and acceptance reasoning but do not prove deployment safety by label alone.",
            "E2": "Acceptance criteria and evidence must match the operational context and intended system behavior.",
            "E3": "This card is not runtime-controller evidence and should not be used as a fallback or hard-stop mechanism.",
        },
    ),
    DistilledCardSpec(
        canonical_id="M05_DataSetShift-Sinha-2022-SystemLevelOODRobotics.pdf",
        filename_stem="M05_DataSetShift-Sinha-2022-SystemLevelOODRobotics.pdf",
        compiler_role="system-level OOD / testing-evaluation / scope-transfer card",
        use_when=[
            "A proposal or judge output transfers trust from one scenario, dataset, or operating context to another.",
            "The compiler must reason about system-level OOD, downstream competence, or functional uncertainty.",
            "A weak model may overload on evidence and ignore scope-transfer limits.",
        ],
        do_not_use_when=[
            "The required evidence is runtime fallback-controller design.",
            "The card would be used as a validated hard-stop algorithm.",
            "The proposal has no dataset, environment, system-level behavior, or transfer issue.",
        ],
        directly_supported_claims=[
            "Robotics OOD should be treated at the system level, not only as isolated classifier uncertainty.",
            "Distribution shift can affect downstream competence and task outcomes.",
            "Testing evidence is limited by scenario coverage and transfer assumptions.",
            "A model can be confident locally while the system behaves unreliably under shifted conditions.",
            "Scope transfer requires additional validation rather than silent trust transfer.",
        ],
        scope_conditions=[
            "Training/evaluation context must be compared to intended deployment context.",
            "Downstream task effects must be considered, not only perception accuracy.",
            "Scenario coverage and shift type should be explicit.",
            "OOD evidence should be tied to system behavior.",
            "Claims should avoid extrapolating beyond tested conditions.",
        ],
        not_supported=[
            "A certified runtime monitor or fallback controller.",
            "A universal OOD detector for all robotics contexts.",
            "Deployment approval from dataset-level performance alone.",
            "Hard-stop propagation unless paired with explicit compiler rules.",
            "Using generic shift language as proof of safety.",
        ],
        weak_reader_traps=[
            "Treating OOD detection as complete system assurance.",
            "Ignoring downstream effects after perception errors.",
            "Using long evidence packets that bury the scope-transfer warning.",
            "Confusing test coverage with deployment validity.",
        ],
        backend_policy=[
            "Mistral: state that scope transfer cannot soften reject/unacceptable.",
            "GPT-3.5: keep packet to 2-3 scope-transfer snippets.",
            "GPT-4.1-nano: separate real unsupported claims from cautious scope caveats.",
            "Grok: require evidence IDs for OOD and transfer-limit claims.",
        ],
        evidence_packet_snippets={
            "E1": "System-level OOD means shift must be judged by downstream robotic competence, not only isolated model confidence.",
            "E2": "Testing evidence transfers only within the evaluated scenario coverage and operating assumptions.",
            "E3": "Sinha does not provide a validated runtime monitor, fallback controller, or deployment approval rule.",
        },
    ),
    DistilledCardSpec(
        canonical_id="M01-02_Introduction-HernandezOrallo-2019-SurveyingSafetyRelevantAI-pdf",
        filename_stem="M01-02_Introduction-HernandezOrallo-2019-SurveyingSafetyRelevantAI-pdf",
        compiler_role="safety-relevant AI taxonomy/framing card",
        use_when=[
            "The compiler needs broad framing for safety-relevant AI characteristics and evaluation limits.",
            "A model needs help distinguishing safety-relevant properties from ordinary performance metrics.",
            "Operator intervention, autonomy level, testing, or monitoring is discussed conceptually.",
        ],
        do_not_use_when=[
            "Direct runtime hard-stop, fallback-controller, or safety-envelope evidence is required.",
            "The card would be used to approve or reject by itself.",
            "The question needs concrete V&V, OOD, or safety-case evidence rather than taxonomy.",
        ],
        directly_supported_claims=[
            "Safety-relevant AI can be organized by characteristics that affect evaluation and assurance.",
            "Performance metrics alone do not capture all safety-relevant properties.",
            "Operator intervention and monitoring are safety-relevant properties, not automatic containment.",
            "Evaluation difficulty depends on scope, environment, and system behavior.",
            "Taxonomy helps route evidence but does not settle assurance claims.",
        ],
        scope_conditions=[
            "Use as framing for routing and vocabulary, not as primary proof.",
            "Pair with specific evidence cards for claims about OOD, runtime assurance, or safety cases.",
            "Keep intervention, oversight, and containment conceptually separate.",
            "Use when proposal wording is ambiguous and needs classification.",
            "Do not infer mechanisms absent from the proposal or cited evidence.",
        ],
        not_supported=[
            "Direct hard-stop evidence.",
            "Runtime monitor or fallback-controller design.",
            "Deployment approval.",
            "Proof that human oversight is containment.",
            "Replacement for domain-specific V&V or safety-case evidence.",
        ],
        weak_reader_traps=[
            "Using taxonomy as if it were an assurance case.",
            "Treating operator intervention as sufficient containment.",
            "Overgeneralizing from conceptual categories to implementation guarantees.",
            "Letting broad safety vocabulary obscure missing evidence.",
        ],
        backend_policy=[
            "Mistral: use only for framing before hard-stop evidence.",
            "GPT-3.5: include one taxonomy snippet plus one specific evidence card.",
            "GPT-4.1-nano: use to calibrate metric/caveat language.",
            "Grok: require evidence IDs when taxonomy supports routing decisions.",
        ],
        evidence_packet_snippets={
            "E1": "Hernandez-Orallo supports safety-relevant AI taxonomy and routing, not direct hard-stop proof.",
            "E2": "Operator intervention is safety-relevant but must not be confused with containment.",
            "E3": "This card should be paired with specific OOD, runtime-assurance, or safety-case evidence before making assurance decisions.",
        },
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create compiler-ready distilled versions of the five principal repaired cards.")
    parser.add_argument("--repaired-dir", default=str(REPAIRED_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--index", default=str(INDEX_PATH))
    parser.add_argument("--audit", default=str(AUDIT_PATH))
    parser.add_argument("--json", default=str(JSON_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repaired_dir = Path(args.repaired_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = generate_distilled_cards(repaired_dir=repaired_dir, output_dir=output_dir)
    Path(args.index).write_text(render_index(cards), encoding="utf-8")
    Path(args.audit).write_text(render_quality_audit(cards), encoding="utf-8")
    Path(args.json).write_text(json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_dir}")
    print(f"Wrote {args.index}")
    print(f"Wrote {args.audit}")
    print(f"Wrote {args.json}")
    return 0


def generate_distilled_cards(*, repaired_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for spec in DISTILLED_SPECS:
        source = find_repaired_card(repaired_dir, spec.filename_stem)
        if source is None:
            raise SystemExit(f"Missing repaired source for {spec.canonical_id}")
        text = render_card(spec, source)
        out_path = output_dir / f"{spec.filename_stem}.distilled.md"
        out_path.write_text(text, encoding="utf-8")
        audit = audit_distilled_card(text, spec)
        row = asdict(spec)
        row.update({
            "source_repaired_card": str(source),
            "distilled_card_path": str(out_path),
            "word_count": word_count(text),
            "quality": audit,
        })
        cards.append(row)
    return cards


def find_repaired_card(repaired_dir: Path, filename_stem: str) -> Path | None:
    exact = repaired_dir / f"{filename_stem}.repaired.md"
    if exact.exists():
        return exact
    matches = sorted(repaired_dir.glob(f"*{filename_stem}*.repaired.md"))
    return matches[0] if matches else None


def render_card(spec: DistilledCardSpec, source: Path) -> str:
    lines = [
        f"# Distilled Principal Card: {spec.canonical_id}",
        "",
        "## Canonical ID",
        spec.canonical_id,
        "",
        "## Compiler role",
        spec.compiler_role,
        "",
        "## Source",
        f"- Derived from: `{source}`",
        "- Original repaired card was not modified.",
        "",
        "## Use when",
        *bullets(spec.use_when),
        "",
        "## Do not use when",
        *bullets(spec.do_not_use_when),
        "",
        "## Directly supported claims",
        *bullets(spec.directly_supported_claims[:5]),
        "",
        "## Scope conditions",
        *bullets(spec.scope_conditions[:5]),
        "",
        "## Not supported by this paper",
        *bullets(spec.not_supported[:5]),
        "",
        "## Weak-reader traps",
        *bullets(spec.weak_reader_traps[:4]),
        "",
        "## Backend policy",
        *bullets(spec.backend_policy[:4]),
        "",
        "## Evidence packet snippets",
        f"- E1: {spec.evidence_packet_snippets['E1']}",
        f"- E2: {spec.evidence_packet_snippets['E2']}",
        f"- E3: {spec.evidence_packet_snippets['E3']}",
        "",
    ]
    return "\n".join(lines)


def bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items]


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def audit_distilled_card(text: str, spec: DistilledCardSpec) -> dict[str, Any]:
    lower = text.lower()
    errors: list[str] = []
    if word_count(text) > 500:
        errors.append("over_500_words")
    if "## Do not use when" not in text:
        errors.append("missing_do_not_use_when")
    if "openai_primary" in lower or "mistral_weak_reader" in lower or "grok_critique" in lower or "provider notes" in lower:
        errors.append("provider_transcript_leak")
    if spec.compiler_role == "runtime-assurance hard-stop card" and "soter" not in spec.canonical_id.lower():
        errors.append("non_soter_runtime_hard_stop_role")
    if spec.compiler_role != "runtime-assurance hard-stop card" and "runtime-assurance hard-stop card" in spec.compiler_role:
        errors.append("invalid_runtime_role")
    if spec.compiler_role.startswith("safety-case") or "safety-case" in spec.compiler_role or "V&V" in spec.compiler_role:
        forbidden = ["role\nruntime-assurance hard-stop", "fallback-controller design.\n## Compiler role"]
        if any(item in lower for item in forbidden):
            errors.append("safety_case_claims_runtime_controller_role")
    return {
        "accepted": not errors,
        "errors": errors,
        "word_count": word_count(text),
        "has_do_not_use_when": "## Do not use when" in text,
        "provider_transcript_free": not any(term in lower for term in ["openai_primary", "mistral_weak_reader", "grok_critique", "provider notes"]),
    }


def render_index(cards: list[dict[str, Any]]) -> str:
    lines = [
        "# Principal Distilled Cards Index",
        "",
        "No API calls were made. These compact cards are derived from the full repaired cards and are intended for compiler/RAG payloads.",
        "",
        "| Canonical ID | Compiler role | Words | Accepted | Distilled card |",
        "|---|---|---:|---:|---|",
    ]
    for card in cards:
        lines.append(
            f"| {card['canonical_id']} | {card['compiler_role']} | {card['word_count']} | "
            f"{card['quality']['accepted']} | `{card['distilled_card_path']}` |"
        )
    return "\n".join(lines) + "\n"


def render_quality_audit(cards: list[dict[str, Any]]) -> str:
    lines = [
        "# Principal Distilled Cards Quality Audit",
        "",
        "No API calls were made. Original repaired cards were not modified.",
        "",
        "| Card | Role | Words | Under 500 | Has Do not use when | Transcript-free | Accepted | Errors |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for card in cards:
        quality = card["quality"]
        lines.append(
            f"| {card['canonical_id']} | {card['compiler_role']} | {quality['word_count']} | "
            f"{quality['word_count'] <= 500} | {quality['has_do_not_use_when']} | "
            f"{quality['provider_transcript_free']} | {quality['accepted']} | "
            f"{'; '.join(quality['errors']) if quality['errors'] else 'none'} |"
        )
    soter_runtime = [
        card for card in cards
        if card["compiler_role"] == "runtime-assurance hard-stop card"
    ]
    lines.extend([
        "",
        "## Role Boundary Checks",
        "",
        f"- SOTER is the only runtime-assurance hard-stop card: {len(soter_runtime) == 1 and 'SOTER' in soter_runtime[0]['canonical_id']}",
        "- Safety-case cards are evidence-discipline cards and do not claim to be runtime controllers.",
        "- Sinha is scoped to OOD/testing and scope-transfer risk, not runtime monitor design.",
        "- Hernandez-Orallo is scoped to taxonomy/framing, not direct hard-stop evidence.",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
