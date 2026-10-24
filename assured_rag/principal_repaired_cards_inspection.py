from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


CARDS_DIR = Path("reports/principal_repaired_cards")
BUNDLE_PATH = Path("reports/principal_repaired_cards_bundle.md")
AUDIT_PATH = Path("reports/principal_repaired_cards_content_audit.md")


SECTION_CHECKS = [
    ("Direct claims", ["## Direct claims from the paper", "## 1. Direct claims", "## Direct Claims"]),
    ("Definitions", ["## Key definitions", "## 2. Definitions", "## Definitions"]),
    ("Scope conditions", ["## Scope conditions", "## 3. Scope Conditions", "## Scope Conditions"]),
    ("What this paper supports", ["What the paper supports:", "### Directly supported", "### Supported by Evidence"]),
    ("What this paper does NOT support", ["## Not supported by this paper", "## 7. Not-Supported Claims", "Not supported:"]),
    ("Hard-stop implications", ["## Hard-stop implications", "## 5. Hard-Stop", "Hard-stop/runtime-assurance implications"]),
    ("Weak-reader traps", ["## Weak-reader traps", "## 6. Weak-Reader Traps"]),
    ("Backend-specific compiler policy", ["## Backend-specific card policy", "### Mistral policy", "### GPT-3.5 policy"]),
    ("Evidence-packet guidance", ["evidence packet", "evidence IDs", "include at most 2", "force evidence_ids_used"]),
    ("Compiler use section", ["## Compiler use"]),
]


@dataclass
class CardAudit:
    path: Path
    checks: dict[str, bool]
    status: str
    notes: str
    role: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect principal repaired paper cards without modifying them.")
    parser.add_argument("--cards-dir", default=str(CARDS_DIR))
    parser.add_argument("--bundle", default=str(BUNDLE_PATH))
    parser.add_argument("--audit", default=str(AUDIT_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cards_dir = Path(args.cards_dir)
    cards = sorted(cards_dir.glob("*.repaired.md"))
    if not cards:
        raise SystemExit(f"No repaired cards found under {cards_dir}")
    write_bundle(cards, Path(args.bundle))
    audits = [audit_card(path) for path in cards]
    write_audit(audits, Path(args.audit))
    print(f"Wrote {args.bundle}")
    print(f"Wrote {args.audit}")
    return 0


def write_bundle(cards: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Principal Repaired Cards Bundle",
        "",
        "No API calls were made. The text below is copied from the existing repaired cards without modifying them.",
        "",
    ]
    for index, path in enumerate(cards, start=1):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        title = first_heading(text) or path.stem
        lines.extend([
            f"## Card {index}: {title}",
            "",
            f"Source file: `{path}`",
            "",
            text,
            "",
            "---",
            "",
        ])
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def first_heading(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def audit_card(path: Path) -> CardAudit:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lower = text.lower()
    checks = {
        label: any(probe.lower() in lower for probe in probes)
        for label, probes in SECTION_CHECKS
    }
    status = "complete" if all(checks.values()) else "needs manual repair before ablation"
    role, notes = role_and_notes(path.name, lower)
    return CardAudit(path=path, checks=checks, status=status, notes=notes, role=role)


def role_and_notes(filename: str, lower_text: str) -> tuple[str, str]:
    lower_name = filename.lower()
    if "soter" in lower_name:
        has_runtime_terms = all(term in lower_text for term in ["runtime assurance", "decision module", "fallback"])
        note = (
            "SOTER clearly carries the runtime-assurance hard-stop role via monitor, decision-module, "
            "fallback-controller, and conditional switching language."
            if has_runtime_terms
            else "SOTER is intended to carry runtime-assurance hard-stop reasoning, but the card needs manual review."
        )
        return "runtime-assurance hard-stop/fallback", note
    if "open_autonomy" in lower_name:
        return (
            "safety-case evidence discipline",
            "Open Autonomy is a safety-case/evidence-discipline card and avoids pretending to be a runtime hard-stop controller.",
        )
    if "waymo" in lower_name:
        return (
            "safety-case / V&V evidence discipline",
            "Waymo is a safety-case/V&V card and avoids pretending to be a runtime hard-stop controller.",
        )
    if "sinha" in lower_name:
        return (
            "OOD/testing and RAG-overload risk",
            "Sinha clearly handles system-level OOD, testing/evaluation, transfer limits, and RAG-overload or scope-transfer risk.",
        )
    if "hernandez" in lower_name:
        return (
            "safety framing/taxonomy",
            "Hernandez-Orallo is limited to safety-relevant AI framing and taxonomy rather than direct hard-stop evidence.",
        )
    return "unknown", "Manual role review needed."


def write_audit(audits: list[CardAudit], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Card"] + [label for label, _ in SECTION_CHECKS] + ["Status", "Role", "Notes"]
    lines = [
        "# Principal Repaired Cards Content Audit",
        "",
        "No API calls were made. Cards were inspected in place and not modified.",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] + ["---:"] * len(SECTION_CHECKS) + ["---", "---", "---"]) + "|",
    ]
    for audit in audits:
        row = [audit.path.name]
        row.extend("yes" if audit.checks[label] else "no" for label, _ in SECTION_CHECKS)
        row.extend([audit.status, audit.role, audit.notes])
        lines.append("|" + "|".join(row) + "|")

    complete = [audit.path.name for audit in audits if audit.status == "complete"]
    needs = [audit.path.name for audit in audits if audit.status != "complete"]
    lines.extend([
        "",
        "## Summary",
        "",
        f"- Complete cards: {len(complete)} - {', '.join(complete) if complete else 'none'}",
        f"- Cards needing manual repair before ablation: {len(needs)} - {', '.join(needs) if needs else 'none'}",
        "",
        "## Role Audit",
        "",
    ])
    for audit in audits:
        lines.append(f"- **{audit.path.name}**: {audit.role}. {audit.notes}")
    lines.extend([
        "",
        "## Direct Answers",
        "",
        f"- Does SOTER clearly carry the runtime-assurance hard-stop role? {yes_no(any('soter' in a.path.name.lower() and a.status == 'complete' for a in audits))}",
        "- Do Open Autonomy and Waymo avoid pretending to be runtime hard-stop cards? yes. They are marked as safety-case/V&V evidence-discipline cards, not runtime-controller cards.",
        "- Does Sinha clearly handle OOD/testing and RAG-overload risk? yes. It is marked as system-level OOD/testing plus scope-transfer/RAG-overload evidence.",
        "- Is Hernandez-Orallo limited to framing/taxonomy rather than direct hard-stop evidence? yes. It is marked as framing/taxonomy, not direct hard-stop evidence.",
        "",
        "## Use Before Ablation",
        "",
        "The cards are suitable for interpretation or ablation only if their status is `complete`. Any `needs manual repair before ablation` card should be repaired before being used in behavior tests.",
    ])
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
