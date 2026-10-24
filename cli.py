"""Command-line interface for AATE.

Examples:

    # 1. Build the index once
    python -m aate ingest --corpus papers/ --index-dir .aate_index

    # 2. Analyze in one mode
    python -m aate analyze --index-dir .aate_index --mode rag \\
        --input some_advisory.pdf --provider anthropic

    # 3. Compare all three modes on the same input
    python -m aate compare --index-dir .aate_index \\
        --input https://www.cisa.gov/news-events/cybersecurity-advisories/aaXX-NNN \\
        --provider anthropic --out report.md
"""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from .ingest import ingest_course_corpus
from .indexing import build_index, RagIndex
from .loaders import load_new_item
from .analyze import analyze_baseline, analyze_with_rag, analyze_with_rubric
from .compare import compare_modes, render_comparison_markdown
from .concept_debate import (
    ConceptDebateConfig,
    apply_concepts_to_policy,
    run_concept_debate,
)
from .corpus_recommender import (
    recommend_corpus,
    render_markdown as render_corpus_recommendations,
    save_report as save_corpus_recommendations,
)
from .data_access_batch import run_batch as run_data_access_batch
from .case_concept_loop import CaseConceptLoopConfig, run_case_concept_loop
from .semantic_sweep import main as semantic_sweep_main
from .techtrack_synthesis import main as techtrack_synthesis_main
from .external_papers import main as external_papers_main
from .deep_pdf_search import main as deep_pdf_search_main
from .full_reading_sweep import main as full_reading_sweep_main
from .sira_sweep import main as sira_sweep_main
from .sira_card_curator import main as sira_card_curator_main
from .card_curator import main as card_curator_main
from .bridge_graph_rag import main as bridge_graph_rag_main
from .council_bridge_rag import main as council_bridge_rag_main
from .concept_graph import main as concept_graph_main
from .assurance_case_cards import main as assurance_case_cards_main
from .reading_map import main as reading_map_main
from .rapid_classifier import main as rapid_classifier_main
from .corpus_recsys import main as corpus_recsys_main
from .agentic_rag import AgenticRagAgent, AgenticRagConfig
from .evaluate_agent import EVAL_SUITES, compare_reports, rescore_report, run_eval
from .fine_tuning_dataset import build_instruction_dataset
from .iterative_rag import IterativeRagConfig, IterativeRagRunner
from .llm import get_llm
from .policy_debate import DebaterConfig, PolicyDebateConfig, run_policy_debate
from .policy_rag import DEFAULT_POLICY_QUESTION, PolicyRagConfig, run_policy_rag
from .prompt_router import route_prompt
from .rubric import rubric_table


def _cmd_ingest(args: argparse.Namespace) -> int:
    chunks = ingest_course_corpus(args.corpus)
    if not chunks:
        print("No chunks ingested. Is the directory layout correct?", file=sys.stderr)
        return 2
    index = build_index(chunks)
    out = Path(args.index_dir)
    index.save(out)
    print(f"Saved index to {out} ({len(chunks)} chunks).")
    return 0


def _load_index(index_dir: str) -> RagIndex:
    return RagIndex.load(index_dir)


def _read_input_arg(args: argparse.Namespace) -> str:
    if args.input == "-":
        return sys.stdin.read()
    return args.input


def _cmd_analyze(args: argparse.Namespace) -> int:
    item = load_new_item(_read_input_arg(args))
    llm = get_llm(args.provider, model=args.model)
    if args.mode == "baseline":
        result = analyze_baseline(item, llm)
    else:
        index = _load_index(args.index_dir)
        if args.mode == "rag":
            result = analyze_with_rag(item, index, llm)
        elif args.mode == "rag_rubric":
            result = analyze_with_rubric(item, index, llm)
        else:
            print(f"Unknown mode: {args.mode}", file=sys.stderr)
            return 2
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.raw_output)
        if result.rubric_scores:
            print("\n\n" + rubric_table(result.rubric_scores))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    item = load_new_item(_read_input_arg(args))
    index = _load_index(args.index_dir)
    llm = get_llm(args.provider, model=args.model)
    judge = (get_llm(args.judge_provider, model=args.judge_model)
             if args.judge_provider else None)
    report = compare_modes(item, index, llm, judge_llm=judge)
    md = render_comparison_markdown(report)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(md)
    if args.json:
        json_path = (args.out or "report") + ".json"
        Path(json_path).write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"Wrote {json_path}")
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    count = build_instruction_dataset(
        args.papers_dir,
        args.out,
        expanded=args.expanded,
        max_chunks_per_module=args.max_chunks_per_module,
    )
    print(f"Wrote {count} instruction examples to {args.out}")
    return 0


def _cmd_agent(args: argparse.Namespace) -> int:
    cfg = AgenticRagConfig(
        index_dir=Path(args.index_dir),
        papers_dir=Path(args.papers_dir),
        provider=args.provider,
        model=args.model,
        top_k=args.top_k,
        include_optional=args.include_optional,
        route_prompt=args.route_prompt,
    )
    agent = AgenticRagAgent(cfg)
    print(agent.answer(args.query))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    cfg = AgenticRagConfig(
        index_dir=Path(args.index_dir),
        papers_dir=Path(args.papers_dir),
        provider=args.provider,
        model=args.model,
        top_k=args.top_k,
        include_optional=args.include_optional,
        route_prompt=args.route_prompt,
    )
    run_eval(cfg, out_dir=args.out_dir, cases=EVAL_SUITES[args.suite])
    return 0


def _cmd_eval_rescore(args: argparse.Namespace) -> int:
    for report in args.reports:
        rescore_report(report)
    if len(args.reports) > 1:
        compare_reports(args.reports, out_path=args.out)
    return 0


def _cmd_inspect_route(args: argparse.Namespace) -> int:
    routed = route_prompt(args.query)
    print(json.dumps(routed.to_dict(), indent=2))
    return 0


def _cmd_iterative_agent(args: argparse.Namespace) -> int:
    cfg = IterativeRagConfig(
        index_dir=Path(args.index_dir),
        papers_dir=Path(args.papers_dir),
        provider=args.provider,
        model=args.model,
        top_k_per_probe=args.top_k_per_probe,
        max_probes=args.max_probes,
        synthesis_max_tokens=args.synthesis_max_tokens,
        revise=not args.no_revise,
        out_dir=Path(args.out_dir),
    )
    result = IterativeRagRunner(cfg).run(args.query)
    print(result["answer"])
    return 0


def _cmd_policy_rag(args: argparse.Namespace) -> int:
    question = args.question
    if args.question_file:
        question = Path(args.question_file).read_text(encoding="utf-8")
    out_dir = _unique_policy_run_dir(Path(args.out_dir), args.provider, args.model)
    iterative_cfg = IterativeRagConfig(
        index_dir=Path(args.index_dir),
        papers_dir=Path(args.papers_dir),
        provider=args.provider,
        model=args.model,
        top_k_per_probe=args.top_k_per_probe,
        max_probes=args.max_probes,
        synthesis_max_tokens=args.synthesis_max_tokens,
        revise=not args.no_revise,
        out_dir=out_dir,
    )
    result = run_policy_rag(
        cfg=PolicyRagConfig(iterative=iterative_cfg, max_policy_chars=args.max_policy_chars),
        policy_urls=args.policy_url,
        policy_file=args.policy_file,
        policy_text=args.policy_text,
        question=question,
    )
    print(result["answer"])
    print(f"\nSaved run directory: {out_dir}")
    return 0


def _cmd_policy_debate(args: argparse.Namespace) -> int:
    question = args.question
    if args.question_file:
        question = Path(args.question_file).read_text(encoding="utf-8")
    cfg = PolicyDebateConfig(
        debater_a=DebaterConfig(
            provider=args.a_provider,
            model=args.a_model,
            role=args.a_role,
        ),
        debater_b=DebaterConfig(
            provider=args.b_provider,
            model=args.b_model,
            role=args.b_role,
        ),
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        index_dir=Path(args.index_dir),
        papers_dir=Path(args.papers_dir),
        aa_top_k_per_concept=args.aa_top_k_per_concept,
        max_policy_chars=args.max_policy_chars,
        max_tokens=args.max_tokens,
        stage_retries=args.stage_retries,
        out_dir=Path(args.out_dir),
    )
    result = run_policy_debate(
        cfg=cfg,
        policy_urls=args.policy_url,
        policy_file=args.policy_file,
        policy_text=args.policy_text,
        question=question,
    )
    print(result["judgment"])
    return 0


def _cmd_concept_debate(args: argparse.Namespace) -> int:
    cfg = ConceptDebateConfig(
        index_dir=Path(args.index_dir),
        papers_dir=Path(args.papers_dir),
        provider_a=args.a_provider,
        model_a=args.a_model,
        provider_b=args.b_provider,
        model_b=args.b_model,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stage_retries=args.stage_retries,
        out_dir=Path(args.out_dir),
    )
    result = run_concept_debate(cfg, args.question)
    print(result["judgment"])
    return 0


def _cmd_apply_concepts(args: argparse.Namespace) -> int:
    question = args.question
    if args.question_file:
        question = Path(args.question_file).read_text(encoding="utf-8")
    result = apply_concepts_to_policy(
        concept_run=Path(args.concept_run),
        policy_urls=args.policy_url,
        policy_file=args.policy_file,
        policy_text=args.policy_text,
        provider=args.provider,
        model=args.model,
        question=question,
        max_policy_chars=args.max_policy_chars,
        max_tokens=args.max_tokens,
        out_dir=Path(args.out_dir),
    )
    print(result["answer"])
    return 0


def _cmd_data_access_batch(args: argparse.Namespace) -> int:
    return run_data_access_batch(args)


def _cmd_case_concept_loop(args: argparse.Namespace) -> int:
    cfg = CaseConceptLoopConfig(
        case_run=Path(args.case_run),
        extractor_provider=args.extractor_provider,
        extractor_model=args.extractor_model,
        provider_a=args.a_provider,
        model_a=args.a_model,
        provider_b=args.b_provider,
        model_b=args.b_model,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        index_dir=Path(args.index_dir),
        papers_dir=Path(args.papers_dir),
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        extractor_max_tokens=args.extractor_max_tokens,
        stage_retries=args.stage_retries,
        max_case_chars=args.max_case_chars,
        out_dir=Path(args.out_dir),
    )
    run_case_concept_loop(cfg)
    return 0


def _cmd_semantic_sweep(args: argparse.Namespace) -> int:
    sweep_args = []
    for key, value in vars(args).items():
        if key in {"func", "cmd"} or value is None or value is False:
            continue
        opt = "--" + key.replace("_", "-")
        if value is True:
            sweep_args.append(opt)
        elif isinstance(value, list):
            if value:
                sweep_args.append(opt)
                sweep_args.extend(str(v) for v in value)
        else:
            sweep_args.extend([opt, str(value)])
    return semantic_sweep_main(sweep_args)


def _cmd_techtrack_synthesis(args: argparse.Namespace) -> int:
    synth_args = []
    for key, value in vars(args).items():
        if key in {"func", "cmd"} or value is None or value is False:
            continue
        opt = "--" + key.replace("_", "-")
        if value is True:
            synth_args.append(opt)
        elif isinstance(value, list):
            if value:
                synth_args.append(opt)
                synth_args.extend(str(v) for v in value)
        else:
            synth_args.extend([opt, str(value)])
    return techtrack_synthesis_main(synth_args)


def _cmd_fetch_external_papers(args: argparse.Namespace) -> int:
    fetch_args = []
    for key, value in vars(args).items():
        if key in {"func", "cmd"} or value is None or value is False:
            continue
        opt = "--" + key.replace("_", "-")
        if value is True:
            fetch_args.append(opt)
        elif isinstance(value, list):
            if value:
                fetch_args.append(opt)
                fetch_args.extend(str(v) for v in value)
        else:
            fetch_args.extend([opt, str(value)])
    return external_papers_main(fetch_args)


def _cmd_deep_pdf_search(args: argparse.Namespace) -> int:
    search_args = []
    for key, value in vars(args).items():
        if key in {"func", "cmd"} or value is None or value is False:
            continue
        opt = "--" + key.replace("_", "-")
        if value is True:
            search_args.append(opt)
        elif isinstance(value, list):
            if value:
                for v in value:
                    search_args.extend([opt, str(v)])
        else:
            search_args.extend([opt, str(value)])
    return deep_pdf_search_main(search_args)


def _cmd_full_reading_sweep(args: argparse.Namespace) -> int:
    sweep_args = []
    for key, value in vars(args).items():
        if key in {"func", "cmd"} or value is None or value is False:
            continue
        opt = "--" + key.replace("_", "-")
        if value is True:
            sweep_args.append(opt)
        elif isinstance(value, list):
            if value:
                for v in value:
                    sweep_args.extend([opt, str(v)])
        else:
            sweep_args.extend([opt, str(value)])
    return full_reading_sweep_main(sweep_args)


def _cmd_sira_sweep(args: argparse.Namespace) -> int:
    sweep_args = []
    for key, value in vars(args).items():
        if key in {"func", "cmd"} or value is None or value is False:
            continue
        opt = "--" + key.replace("_", "-")
        if value is True:
            sweep_args.append(opt)
        elif isinstance(value, list):
            if value:
                for v in value:
                    sweep_args.extend([opt, str(v)])
        else:
            sweep_args.extend([opt, str(value)])
    return sira_sweep_main(sweep_args)


def _cmd_curate_cards(args: argparse.Namespace) -> int:
    curator_args = []
    for key, value in vars(args).items():
        if key in {"func", "cmd"} or value is None or value is False:
            continue
        opt = "--" + key.replace("_", "-")
        if value is True:
            curator_args.append(opt)
        else:
            curator_args.extend([opt, str(value)])
    return card_curator_main(curator_args)


def _cmd_curate_sira_cards(args: argparse.Namespace) -> int:
    curator_args = []
    for key, value in vars(args).items():
        if key in {"func", "cmd"} or value is None or value is False:
            continue
        opt = "--" + key.replace("_", "-")
        if value is True:
            curator_args.append(opt)
        else:
            curator_args.extend([opt, str(value)])
    return sira_card_curator_main(curator_args)


def _cmd_bridge_rag(args: argparse.Namespace) -> int:
    bridge_args = []
    for key, value in vars(args).items():
        if key in {"func", "cmd"} or value is None or value is False:
            continue
        opt = "--" + key.replace("_", "-")
        if key == "query":
            bridge_args.append(str(value))
        elif value is True:
            bridge_args.append(opt)
        else:
            bridge_args.extend([opt, str(value)])
    return bridge_graph_rag_main(bridge_args)


def _cmd_council_bridge_rag(args: argparse.Namespace) -> int:
    council_args = []
    for key, value in vars(args).items():
        if key in {"func", "cmd"} or value is None or value is False:
            continue
        if key == "corrective_retrieval" and value is True:
            continue
        opt = "--" + key.replace("_", "-")
        if key == "query":
            council_args.append(str(value))
        elif value is True:
            council_args.append(opt)
        else:
            council_args.extend([opt, str(value)])
    return council_bridge_rag_main(council_args)


def _cmd_concept_graph(args: argparse.Namespace) -> int:
    graph_args = [args.graph_cmd]
    for key, value in vars(args).items():
        if key in {"func", "cmd", "graph_cmd"} or value is None or value is False:
            continue
        opt = "--" + key.replace("_", "-")
        if value is True:
            graph_args.append(opt)
        elif isinstance(value, list):
            for item in value:
                graph_args.extend([opt, str(item)])
        else:
            graph_args.extend([opt, str(value)])
    return concept_graph_main(graph_args)


def _cmd_assurance_cases(args: argparse.Namespace) -> int:
    case_args = [args.case_cmd]
    for key, value in vars(args).items():
        if key in {"func", "cmd", "case_cmd"} or value is None or value is False:
            continue
        opt = "--" + key.replace("_", "-")
        if value is True:
            case_args.append(opt)
        elif isinstance(value, list):
            for item in value:
                case_args.extend([opt, str(item)])
        else:
            case_args.extend([opt, str(value)])
    return assurance_case_cards_main(case_args)


def _cmd_reading_map(args: argparse.Namespace) -> int:
    map_args = []
    for key, value in vars(args).items():
        if key in {"func", "cmd"} or value is None or value is False:
            continue
        opt = "--" + key.replace("_", "-")
        if value is True:
            map_args.append(opt)
        else:
            map_args.extend([opt, str(value)])
    return reading_map_main(map_args)


def _cmd_rapid_classifier(args: argparse.Namespace) -> int:
    classifier_args = []
    for key, value in vars(args).items():
        if key in {"func", "cmd"} or value is None or value is False:
            continue
        opt = "--" + key.replace("_", "-")
        if value is True:
            classifier_args.append(opt)
        elif isinstance(value, list):
            for item in value:
                classifier_args.extend([opt, str(item)])
        else:
            classifier_args.extend([opt, str(value)])
    return rapid_classifier_main(classifier_args)


def _cmd_corpus_recommend(args: argparse.Namespace) -> int:
    query = args.query
    if args.query_file:
        query = Path(args.query_file).read_text(encoding="utf-8")
    if not query:
        print("corpus-recommend requires a query argument or --query-file", file=sys.stderr)
        return 2
    report = recommend_corpus(
        query,
        index_dir=Path(args.index_dir),
        papers_dir=Path(args.papers_dir),
        top_papers=args.top_papers,
        passages_per_paper=args.passages_per_paper,
        candidate_chunks=args.candidate_chunks,
        include_optional=args.include_optional,
        route_top_k=args.route_top_k,
        diversity_weight=args.diversity_weight,
        required_weight=args.required_weight,
        reranker_model=None if args.no_reranker else args.reranker_model,
    )
    md = render_corpus_recommendations(report)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(md.encode(encoding, errors="replace").decode(encoding, errors="replace"))
    if args.out_dir:
        out_dir = _unique_named_run_dir(Path(args.out_dir), "corpus_recommend")
        md_path = save_corpus_recommendations(report, out_dir)
        print(f"Saved corpus recommendation report: {md_path}")
    return 0


def _cmd_corpus_recsys(args: argparse.Namespace) -> int:
    recsys_args = []
    for key, value in vars(args).items():
        if key in {"func", "cmd"} or value is None or value is False:
            continue
        opt = "--" + key.replace("_", "-")
        if key == "query":
            recsys_args.append(str(value))
        elif value is True:
            recsys_args.append(opt)
        elif isinstance(value, list):
            for item in value:
                recsys_args.extend([opt, str(item)])
        else:
            recsys_args.extend([opt, str(value)])
    return corpus_recsys_main(recsys_args)


def _unique_policy_run_dir(base_dir: Path, provider: str, model: str | None) -> Path:
    safe_model = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "-"
        for ch in (model or "default")
    ).strip("-") or "default"
    safe_provider = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "-"
        for ch in provider
    ).strip("-") or "provider"
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = base_dir / f"policy_rag_{safe_provider}_{safe_model}_{stamp}"
    n = 1
    while run_dir.exists():
        run_dir = base_dir / f"policy_rag_{safe_provider}_{safe_model}_{stamp}_{n}"
        n += 1
    return run_dir


def _unique_named_run_dir(base_dir: Path, prefix: str) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = base_dir / f"{prefix}_{stamp}"
    n = 1
    while run_dir.exists():
        run_dir = base_dir / f"{prefix}_{stamp}_{n}"
        n += 1
    return run_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="Ingest corpus and build index")
    pi.add_argument("--corpus", required=True, help="Path to papers/ directory")
    pi.add_argument("--index-dir", default=".aate_index",
                    help="Where to save the FAISS index + metadata")
    pi.set_defaults(func=_cmd_ingest)

    pa = sub.add_parser("analyze", help="Analyze one item in one mode")
    pa.add_argument("--input", required=True,
                    help="PDF path, URL, '-' (stdin), or pasted text")
    pa.add_argument("--mode", choices=["baseline", "rag", "rag_rubric"], required=True)
    pa.add_argument("--index-dir", default=".aate_index")
    pa.add_argument("--provider", default="stub",
                    choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    pa.add_argument("--model", default=None)
    pa.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")
    pa.set_defaults(func=_cmd_analyze)

    pc = sub.add_parser("compare", help="Run all three modes and produce a comparison report")
    pc.add_argument("--input", required=True,
                    help="PDF path, URL, '-' (stdin), or pasted text")
    pc.add_argument("--index-dir", default=".aate_index")
    pc.add_argument("--provider", default="stub",
                    choices=["stub", "anthropic", "openai", "xai", "grok", "mistral"])
    pc.add_argument("--model", default=None)
    pc.add_argument("--judge-provider", default=None,
                    choices=[None, "stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"],
                    help="Optional separate provider for the meta-evaluator")
    pc.add_argument("--judge-model", default=None)
    pc.add_argument("--out", default=None, help="Write markdown report to file")
    pc.add_argument("--json", action="store_true",
                    help="Also write a structured JSON report next to --out")
    pc.set_defaults(func=_cmd_compare)

    pd = sub.add_parser("dataset", help="Generate PEFT instruction-tuning JSONL")
    pd.add_argument("--papers-dir", default='papers')
    pd.add_argument("--out", default='data/aa_instruction_tuning.jsonl')
    pd.add_argument("--expanded", action="store_true",
                    help="Add chunk-level and cross-module task-alignment examples")
    pd.add_argument("--max-chunks-per-module", type=int, default=60)
    pd.set_defaults(func=_cmd_dataset)

    pg = sub.add_parser("agent", help="Ask the agentic RAG assistant")
    pg.add_argument("query", help="Question for the ReAct-style RAG agent")
    pg.add_argument("--index-dir", default=".aate_index")
    pg.add_argument("--papers-dir", default='papers')
    pg.add_argument("--provider", default="stub",
                    choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    pg.add_argument("--model", default=None)
    pg.add_argument("--top-k", type=int, default=8)
    pg.add_argument("--include-optional", action="store_true")
    pg.add_argument("--route-prompt", action="store_true",
                    help="Expand raw prompt into RAG-relevant assurance concepts before retrieval")
    pg.set_defaults(func=_cmd_agent)

    pe = sub.add_parser("eval-agent", help="Run fixed evaluation questions against the agent")
    pe.add_argument("--index-dir", default=".aate_index")
    pe.add_argument("--papers-dir", default='papers')
    pe.add_argument("--provider", default="openai",
                    choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    pe.add_argument("--model", default=None)
    pe.add_argument("--top-k", type=int, default=8)
    pe.add_argument("--include-optional", action="store_true")
    pe.add_argument("--out-dir", default="eval_runs")
    pe.add_argument("--suite", default="basic", choices=sorted(EVAL_SUITES),
                    help="Evaluation suite to run")
    pe.add_argument("--route-prompt", action="store_true")
    pe.set_defaults(func=_cmd_eval)

    pi = sub.add_parser("inspect-route", help="Inspect deterministic prompt routing")
    pi.add_argument("query")
    pi.set_defaults(func=_cmd_inspect_route)

    pir = sub.add_parser("iterative-agent", help="Run decompose/retrieve/synthesize/critique RAG")
    pir.add_argument("query")
    pir.add_argument("--index-dir", default=".aate_index")
    pir.add_argument("--papers-dir", default='papers')
    pir.add_argument("--provider", default="openai",
                     choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    pir.add_argument("--model", default="gpt-4.1")
    pir.add_argument("--top-k-per-probe", type=int, default=4)
    pir.add_argument("--max-probes", type=int, default=6)
    pir.add_argument("--synthesis-max-tokens", type=int, default=6000)
    pir.add_argument("--no-revise", action="store_true")
    pir.add_argument("--out-dir", default="iterative_runs")
    pir.set_defaults(func=_cmd_iterative_agent)

    pp = sub.add_parser("policy-rag", help="Analyze policy URL/file/text through iterative RAG")
    src = pp.add_mutually_exclusive_group(required=True)
    src.add_argument("--policy-url", action="append",
                     help="Policy/source URL. Repeat this option to analyze multiple sources together.")
    src.add_argument("--policy-file")
    src.add_argument("--policy-text")
    pp.add_argument("--question", default=DEFAULT_POLICY_QUESTION)
    pp.add_argument("--question-file",
                    help="Read the policy analysis question/instructions from a UTF-8 text file")
    pp.add_argument("--index-dir", default=".aate_index")
    pp.add_argument("--papers-dir", default='papers')
    pp.add_argument("--provider", default="openai",
                    choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    pp.add_argument("--model", default="gpt-4.1")
    pp.add_argument("--top-k-per-probe", type=int, default=4)
    pp.add_argument("--max-probes", type=int, default=8)
    pp.add_argument("--synthesis-max-tokens", type=int, default=6000)
    pp.add_argument("--max-policy-chars", type=int, default=30000)
    pp.add_argument("--no-revise", action="store_true")
    pp.add_argument("--out-dir", default="policy_runs")
    pp.set_defaults(func=_cmd_policy_rag)

    pdeb = sub.add_parser("policy-debate", help="Run a two-model debate over policy sources")
    src = pdeb.add_mutually_exclusive_group(required=True)
    src.add_argument("--policy-url", action="append",
                     help="Policy/source URL. Repeat this option to analyze multiple sources together.")
    src.add_argument("--policy-file")
    src.add_argument("--policy-text")
    pdeb.add_argument("--question", default=(
        "Can a JHU student realistically obtain JuReko-related data for academic research? "
        "Debate whether the best path is raw corpus access, a narrow subset, derived outputs, "
        "or CAL2Lab interface access. Focus on Germany and treat the British pilot as peripheral."
    ))
    pdeb.add_argument("--question-file",
                      help="Read the debate question/instructions from a UTF-8 text file")
    pdeb.add_argument("--a-provider", default="openai",
                      choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    pdeb.add_argument("--a-model", default="gpt-4.1")
    pdeb.add_argument("--a-role", default=(
        "Cautious governance counsel: argue for a conservative interpretation, "
        "avoid overclaiming, and identify permissions that must be obtained in writing."
    ))
    pdeb.add_argument("--b-provider", default="xai",
                      choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    pdeb.add_argument("--b-model", default="grok-4.20-reasoning")
    pdeb.add_argument("--b-role", default=(
        "Practical access strategist: argue for the most feasible path to obtain useful "
        "research data through interfaces, derived exports, or narrow German-only subsets."
    ))
    pdeb.add_argument("--judge-provider", default="openai",
                      choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    pdeb.add_argument("--judge-model", default="gpt-4.1")
    pdeb.add_argument("--index-dir", default=".aate_index")
    pdeb.add_argument("--papers-dir", default='papers')
    pdeb.add_argument("--aa-top-k-per-concept", type=int, default=3,
                      help="Assured Autonomy evidence passages to retrieve for each concept vector")
    pdeb.add_argument("--max-policy-chars", type=int, default=60000)
    pdeb.add_argument("--max-tokens", type=int, default=5000)
    pdeb.add_argument("--stage-retries", type=int, default=2)
    pdeb.add_argument("--out-dir", default="policy_runs")
    pdeb.set_defaults(func=_cmd_policy_debate)

    cdeb = sub.add_parser("concept-debate", help="Debate Assured Autonomy concepts using only the course corpus")
    cdeb.add_argument("question")
    cdeb.add_argument("--index-dir", default=".aate_index")
    cdeb.add_argument("--papers-dir", default='papers')
    cdeb.add_argument("--a-provider", default="openai",
                      choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    cdeb.add_argument("--a-model", default="gpt-4.1")
    cdeb.add_argument("--b-provider", default="xai",
                      choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    cdeb.add_argument("--b-model", default="grok-4.20-reasoning")
    cdeb.add_argument("--judge-provider", default="openai",
                      choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    cdeb.add_argument("--judge-model", default="gpt-4.1")
    cdeb.add_argument("--top-k", type=int, default=4)
    cdeb.add_argument("--max-tokens", type=int, default=5000)
    cdeb.add_argument("--stage-retries", type=int, default=2)
    cdeb.add_argument("--out-dir", default="concept_runs")
    cdeb.set_defaults(func=_cmd_concept_debate)

    capp = sub.add_parser("apply-concepts-to-policy", help="Apply a concept-debate run to policy URL/file/text")
    src = capp.add_mutually_exclusive_group(required=True)
    src.add_argument("--policy-url", action="append")
    src.add_argument("--policy-file")
    src.add_argument("--policy-text")
    capp.add_argument("--concept-run", required=True, help="Path to concept_debate_*.json")
    capp.add_argument("--question", default=DEFAULT_POLICY_QUESTION)
    capp.add_argument("--question-file")
    capp.add_argument("--provider", default="openai",
                      choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    capp.add_argument("--model", default="gpt-4.1")
    capp.add_argument("--max-policy-chars", type=int, default=60000)
    capp.add_argument("--max-tokens", type=int, default=5000)
    capp.add_argument("--out-dir", default="concept_runs")
    capp.set_defaults(func=_cmd_apply_concepts)

    dab = sub.add_parser("data-access-batch", help="Apply AA concepts to several restricted dataset access cases")
    dab.add_argument("--concept-run", required=True, help="Path to concept_debate_*.json")
    dab.add_argument("--cases", default="data/restricted_dataset_access_cases.json")
    dab.add_argument("--template", default="prompts/data_access_assurance_template.txt")
    dab.add_argument("--only", nargs="*", help="Optional case ids to run")
    dab.add_argument("--provider", default="xai",
                     choices=["openai", "xai", "grok", "mistral", "anthropic", "ollama", "stub", "peft-local"])
    dab.add_argument("--model", default="grok-4.20-reasoning")
    dab.add_argument("--max-policy-chars", type=int, default=50000)
    dab.add_argument("--max-tokens", type=int, default=3500)
    dab.add_argument("--min-tokens", type=int, default=1400)
    dab.add_argument("--retries", type=int, default=2)
    dab.add_argument("--retry-sleep", type=float, default=2.0)
    dab.add_argument("--out-dir", default="concept_runs")
    dab.set_defaults(func=_cmd_data_access_batch)

    ccl = sub.add_parser("case-concept-loop",
                         help="Extract corpus-dependent questions from a case run, then run a concept debate")
    ccl.add_argument("--case-run", required=True,
                     help="Path to a policy_debate/policy_rag JSON file or its saved run directory")
    ccl.add_argument("--extractor-provider", default="openai",
                     choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    ccl.add_argument("--extractor-model", default="gpt-4.1")
    ccl.add_argument("--a-provider", default="openai",
                     choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    ccl.add_argument("--a-model", default="gpt-4.1")
    ccl.add_argument("--b-provider", default="xai",
                     choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    ccl.add_argument("--b-model", default="grok-4.20-reasoning")
    ccl.add_argument("--judge-provider", default="openai",
                     choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    ccl.add_argument("--judge-model", default="gpt-4.1")
    ccl.add_argument("--index-dir", default=".aate_index")
    ccl.add_argument("--papers-dir", default='papers')
    ccl.add_argument("--top-k", type=int, default=7)
    ccl.add_argument("--max-tokens", type=int, default=5000)
    ccl.add_argument("--extractor-max-tokens", type=int, default=2500)
    ccl.add_argument("--stage-retries", type=int, default=2)
    ccl.add_argument("--max-case-chars", type=int, default=70000)
    ccl.add_argument("--out-dir", default="concept_runs")
    ccl.set_defaults(func=_cmd_case_concept_loop)

    ss = sub.add_parser("semantic-sweep",
                        help="Build per-paper semantic cards and synthesize a DPRK worker protocol")
    ss.add_argument("--papers-dir", default='papers/papers')
    ss.add_argument("--provider", default="openai",
                    choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    ss.add_argument("--model", default="gpt-5.5")
    ss.add_argument("--max-doc-chars", type=int, default=45000)
    ss.add_argument("--card-max-tokens", type=int, default=2400)
    ss.add_argument("--synthesis-max-tokens", type=int, default=7000)
    ss.add_argument("--sleep-seconds", type=float, default=0.0)
    ss.add_argument("--out-dir", default="semantic_runs")
    ss.add_argument("--modules", nargs="*")
    ss.add_argument("--limit", type=int)
    ss.add_argument("--force", action="store_true")
    ss.add_argument("--synthesize-only", action="store_true")
    ss.add_argument("--case-file")
    ss.add_argument("--case-text")
    ss.add_argument("--case-name", default="dprk_protocol")
    ss.add_argument("--card-retries", type=int, default=2)
    ss.add_argument("--allow-failed-cards", action="store_true")
    ss.set_defaults(func=_cmd_semantic_sweep)

    fep = sub.add_parser("fetch-external-papers",
                         help="Download direct-PDF external reading candidates into papers/external")
    fep.add_argument("--candidates", default="data/external_reading_candidates.json")
    fep.add_argument("--out-dir", default='papers/external')
    fep.add_argument("--only", nargs="*", help="Optional candidate ids to fetch")
    fep.add_argument("--timeout", type=int, default=60)
    fep.add_argument("--overwrite", action="store_true")
    fep.set_defaults(func=_cmd_fetch_external_papers)

    dps = sub.add_parser("deep-pdf-search",
                         help="Search PDF pages directly and save a page-level evidence pack")
    dps.add_argument("--papers-dir", action="append",
                     help="Directory to scan. Repeat for course and external papers.")
    dps.add_argument("--query", required=True)
    dps.add_argument("--out-dir", default="deep_evidence_runs")
    dps.add_argument("--top-pages", type=int, default=30)
    dps.add_argument("--context-chars", type=int, default=900)
    dps.add_argument("--max-pages-per-pdf", type=int, default=500)
    dps.set_defaults(func=_cmd_deep_pdf_search)

    frs = sub.add_parser("full-reading-sweep",
                         help="Comprehensive chunk-by-chunk sweep of readings into study-grade paper cards")
    frs.add_argument("--papers-dir", action="append",
                     help="Directory to scan. Repeat for course and external paper roots.")
    frs.add_argument("--provider", default="openai",
                     choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    frs.add_argument("--model", default="gpt-5.5")
    frs.add_argument("--out-dir", default="full_sweep_runs")
    frs.add_argument("--resume-dir")
    frs.add_argument("--case-file")
    frs.add_argument("--case-text")
    frs.add_argument("--modules", nargs="*")
    frs.add_argument("--limit", type=int)
    frs.add_argument("--chunk-chars", type=int, default=18000)
    frs.add_argument("--chunk-max-tokens", type=int, default=2200)
    frs.add_argument("--paper-max-tokens", type=int, default=5200)
    frs.add_argument("--synthesis-max-tokens", type=int, default=9000)
    frs.add_argument("--retries", type=int, default=2)
    frs.add_argument("--sleep-seconds", type=float, default=0.0)
    frs.add_argument("--force", action="store_true")
    frs.add_argument("--synthesize-only", action="store_true")
    frs.add_argument("--allow-failed-chunks", action="store_true")
    frs.set_defaults(func=_cmd_full_reading_sweep)

    sira = sub.add_parser("sira-sweep",
                          help="SIRA-style DF-filtered retrieval sweep before expensive paper synthesis")
    sira.add_argument("--papers-dir", action="append",
                      help="Directory to scan. Repeat for course and external paper roots.")
    sira.add_argument("--provider", default="openai",
                      choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    sira.add_argument("--model", default="gpt-5.5")
    sira.add_argument("--out-dir", default="sira_sweep_runs")
    sira.add_argument("--resume-dir")
    sira.add_argument("--case-file")
    sira.add_argument("--case-text")
    sira.add_argument("--modules", nargs="*")
    sira.add_argument("--limit", type=int)
    sira.add_argument("--max-pages-per-pdf", type=int, default=500)
    sira.add_argument("--sample-chars", type=int, default=18000)
    sira.add_argument("--top-pages-per-paper", type=int, default=8)
    sira.add_argument("--max-selected-page-chars", type=int, default=18000)
    sira.add_argument("--max-df-ratio", type=float, default=0.45)
    sira.add_argument("--min-df", type=int, default=1)
    sira.add_argument("--query-weight", type=float, default=3.0)
    sira.add_argument("--document-weight", type=float, default=0.85)
    sira.add_argument("--vocab-max-tokens", type=int, default=1600)
    sira.add_argument("--page-card-max-tokens", type=int, default=2600)
    sira.add_argument("--paper-card-max-tokens", type=int, default=4200)
    sira.add_argument("--synthesis-max-tokens", type=int, default=9000)
    sira.add_argument("--retries", type=int, default=2)
    sira.add_argument("--sleep-seconds", type=float, default=0.0)
    sira.add_argument("--force", action="store_true")
    sira.add_argument("--retrieval-only", action="store_true")
    sira.add_argument("--synthesize-only", action="store_true")
    sira.set_defaults(func=_cmd_sira_sweep)

    ccards = sub.add_parser("curate-cards",
                            help="Audit full-sweep paper cards and render readable study notes")
    ccards.add_argument("--card-dir", required=True)
    ccards.add_argument("--out-dir", default="card_curation_runs")
    ccards.set_defaults(func=_cmd_curate_cards)

    scards = sub.add_parser("curate-sira-cards",
                            help="Render compact readable cards from SIRA paper-card JSON files")
    scards.add_argument("--card-dir", required=True)
    scards.add_argument("--out-dir", required=True)
    scards.add_argument("--force", action="store_true")
    scards.set_defaults(func=_cmd_curate_sira_cards)

    br = sub.add_parser("bridge-rag",
                        help="Bridge-Graph RAG with hidden concept cards and model-based query lifting")
    br.add_argument("query", nargs="?", default=None)
    br.add_argument("--query-file")
    br.add_argument("--bridge-dir", default="bridge_cards")
    br.add_argument("--init-seeds", action="store_true")
    br.add_argument("--overwrite-seeds", action="store_true")
    br.add_argument("--provider", default="xai",
                    choices=["xai", "grok", "mistral", "openai", "anthropic", "ollama", "stub", "peft-local"])
    br.add_argument("--model", default="grok-4.20-reasoning")
    br.add_argument("--index-dir", default=".aate_index")
    br.add_argument("--papers-dir", default='papers')
    br.add_argument("--top-papers", type=int, default=14)
    br.add_argument("--passages-per-paper", type=int, default=2)
    br.add_argument("--candidate-chunks", type=int, default=120)
    br.add_argument("--max-bridge-cards", type=int, default=6)
    br.add_argument("--include-optional", action="store_true")
    br.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    br.add_argument("--no-reranker", action="store_true")
    br.add_argument("--lift-max-tokens", type=int, default=2200)
    br.add_argument("--synthesis-max-tokens", type=int, default=3500)
    br.add_argument("--no-synthesis", action="store_true")
    br.add_argument("--out-dir", default="bridge_runs")
    br.set_defaults(func=_cmd_bridge_rag)

    cbr = sub.add_parser("council-bridge-rag",
                         help="Multi-model Bridge RAG: lift, scout, critic, synthesis")
    cbr.add_argument("query", nargs="?", default=None)
    cbr.add_argument("--query-file")
    cbr.add_argument("--bridge-dir", default="bridge_cards")
    cbr.add_argument("--init-seeds", action="store_true")
    cbr.add_argument("--overwrite-seeds", action="store_true")
    cbr.add_argument("--index-dir", default=".aate_index")
    cbr.add_argument("--papers-dir", default='papers')
    cbr.add_argument("--top-papers", type=int, default=10)
    cbr.add_argument("--passages-per-paper", type=int, default=1)
    cbr.add_argument("--candidate-chunks", type=int, default=90)
    cbr.add_argument("--max-bridge-cards", type=int, default=6)
    cbr.add_argument("--concept-graph",
                     help="Optional concept_graph.json built with `aate concept-graph build`")
    cbr.add_argument("--concept-graph-top-k", type=int, default=18)
    cbr.add_argument("--include-optional", action="store_true")
    cbr.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    cbr.add_argument("--no-reranker", action="store_true")
    cbr.add_argument("--lift-provider", default="openai")
    cbr.add_argument("--lift-model", default="gpt-5.5")
    cbr.add_argument("--scout-provider", default="xai")
    cbr.add_argument("--scout-model", default="grok-4.20-reasoning")
    cbr.add_argument("--critic-provider", default="openai")
    cbr.add_argument("--critic-model", default="gpt-4.1")
    cbr.add_argument("--synthesis-provider", default="openai")
    cbr.add_argument("--synthesis-model", default="gpt-5.5")
    cbr.add_argument("--lift-max-tokens", type=int, default=2600)
    cbr.add_argument("--scout-max-tokens", type=int, default=2600)
    cbr.add_argument("--critic-max-tokens", type=int, default=2200)
    cbr.add_argument("--synthesis-max-tokens", type=int, default=3200)
    cbr.add_argument("--no-critic", action="store_true")
    cbr.add_argument("--no-synthesis", action="store_true")
    cbr.add_argument("--no-corrective-retrieval", dest="corrective_retrieval",
                     action="store_false", default=True)
    cbr.add_argument("--gate-only", action="store_true")
    cbr.add_argument("--out-dir", default="council_bridge_runs")
    cbr.set_defaults(func=_cmd_council_bridge_rag)

    cg = sub.add_parser("concept-graph",
                        help="Build/query a persistent assurance concept graph from RAG artifacts")
    cg_sub = cg.add_subparsers(dest="graph_cmd", required=True)
    cgb = cg_sub.add_parser("build", help="Build concept graph JSON from artifacts")
    cgb.add_argument("--artifact-root", action="append", required=True)
    cgb.add_argument("--out-dir", default="concept_graph_runs")
    cgb.add_argument("--max-files", type=int)
    cgb.set_defaults(func=_cmd_concept_graph)
    cgq = cg_sub.add_parser("query", help="Query concept graph JSON")
    cgq.add_argument("--graph", required=True)
    cgq.add_argument("--query", required=True)
    cgq.add_argument("--top-k", type=int, default=20)
    cgq.add_argument("--out")
    cgq.set_defaults(func=_cmd_concept_graph)

    ac = sub.add_parser("assurance-cases",
                        help="Compile council RAG runs into reusable assurance-case cards")
    ac_sub = ac.add_subparsers(dest="case_cmd", required=True)
    acb = ac_sub.add_parser("build", help="Build assurance-case cards from saved council runs")
    acb.add_argument("--run-root", action="append", required=True,
                     help="Council run directory, root directory, or council_bridge_rag.json. Repeatable.")
    acb.add_argument("--provider", default="openai",
                     choices=["openai", "xai", "grok", "mistral", "anthropic", "ollama", "stub", "peft-local"])
    acb.add_argument("--model", default="gpt-5.5")
    acb.add_argument("--max-tokens", type=int, default=5000)
    acb.add_argument("--max-runs", type=int)
    acb.add_argument("--no-llm", action="store_true")
    acb.add_argument("--out-dir", default="assurance_case_runs")
    acb.set_defaults(func=_cmd_assurance_cases)

    rm = sub.add_parser("reading-map",
                        help="Run paper-level PCA/clustering over the reading-list embeddings")
    rm.add_argument("--index-dir", default=".aate_index")
    rm.add_argument("--query", default=(
        "sparse-test trust transfer for autonomous driving; cyber-physical escalation; "
        "scenario coverage; runtime assurance; safe control; dataset shift; adversarial perception"
    ))
    rm.add_argument("--dims", type=int, default=3)
    rm.add_argument("--clusters", type=int, default=10)
    rm.add_argument("--top-query", type=int, default=14)
    rm.add_argument("--top-central", type=int, default=8)
    rm.add_argument("--out-dir", default="reading_map_runs")
    rm.set_defaults(func=_cmd_reading_map)

    rc = sub.add_parser("rapid-classifier",
                        help="Classify repo/code excerpts with RAG module passages into quiz labels")
    rc.add_argument("--repo-root", default="external_repos")
    rc.add_argument("--repo", action="append", help="Specific repo path. Repeatable.")
    rc.add_argument("--index-dir", default=".aate_index")
    rc.add_argument("--provider", default="mistral",
                    choices=["mistral", "openai", "xai", "grok", "anthropic", "ollama", "stub", "peft-local"])
    rc.add_argument("--model", default="mistral-large-latest")
    rc.add_argument("--samples-per-repo", type=int, default=2)
    rc.add_argument("--sequence-mode", action="store_true")
    rc.add_argument("--sequences-per-repo", type=int, default=1)
    rc.add_argument("--snippets-per-sequence", type=int, default=4)
    rc.add_argument("--folds", type=int, default=5)
    rc.add_argument("--quiz", action="store_true")
    rc.add_argument("--quiz-max-tokens", type=int, default=1700)
    rc.add_argument("--max-repos", type=int)
    rc.add_argument("--max-examples", type=int)
    rc.add_argument("--excerpt-chars", type=int, default=2600)
    rc.add_argument("--rag-k", type=int, default=6)
    rc.add_argument("--max-tokens", type=int, default=1400)
    rc.add_argument("--retries", type=int, default=2)
    rc.add_argument("--sleep-seconds", type=float, default=20.0)
    rc.add_argument("--seed", type=int, default=7)
    rc.add_argument("--out-dir", default="rapid_classifier_runs")
    rc.set_defaults(func=_cmd_rapid_classifier)

    cr = sub.add_parser("corpus-recommend",
                        help="Recommend a balanced set of corpus papers/passages for a project question")
    cr.add_argument("query", nargs="?", default=None,
                    help="Project question or assignment need")
    cr.add_argument("--query-file",
                    help="Read the project question/instructions from a UTF-8 text file")
    cr.add_argument("--index-dir", default=".aate_index")
    cr.add_argument("--papers-dir", default='papers')
    cr.add_argument("--top-papers", type=int, default=10)
    cr.add_argument("--passages-per-paper", type=int, default=2)
    cr.add_argument("--candidate-chunks", type=int, default=80)
    cr.add_argument("--route-top-k", type=int, default=6)
    cr.add_argument("--include-optional", action="store_true")
    cr.add_argument("--diversity-weight", type=float, default=0.18)
    cr.add_argument("--required-weight", type=float, default=0.08)
    cr.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    cr.add_argument("--no-reranker", action="store_true")
    cr.add_argument("--out-dir", default="recommendation_runs")
    cr.set_defaults(func=_cmd_corpus_recommend)

    crs = sub.add_parser("corpus-recsys",
                         help="Answer a question and recommend paper-centered corpus discussion paths")
    crs.add_argument("query", nargs="?", default=None,
                     help="Question or project prompt")
    crs.add_argument("--query-file",
                     help="Read the question from a UTF-8 text file")
    crs.add_argument("--index-dir", default=".aate_index")
    crs.add_argument("--papers-dir", default='papers')
    crs.add_argument("--provider", default="mistral",
                     choices=["mistral", "openai", "xai", "grok", "anthropic", "ollama", "stub", "peft-local"])
    crs.add_argument("--model", default="mistral-large-latest")
    crs.add_argument("--mistral-key-file",
                     help="Read the Mistral API key from a local file; avoids embedding secrets in an EXE")
    crs.add_argument("--top-papers", type=int, default=6)
    crs.add_argument("--passages-per-paper", type=int, default=2)
    crs.add_argument("--candidate-chunks", type=int, default=90)
    crs.add_argument("--route-top-k", type=int, default=6)
    crs.add_argument("--include-optional", action="store_true")
    crs.add_argument("--max-tokens", type=int, default=2200)
    crs.add_argument("--no-llm", action="store_true")
    crs.add_argument("--out-dir", default="recsys_runs")
    crs.set_defaults(func=_cmd_corpus_recsys)

    ts = sub.add_parser("techtrack-synthesis",
                        help="Use TechTrack docs plus semantic cards to design an AA experiment plan")
    ts.add_argument("--techtrack-dir",
                    default="external_repos/techtrack")
    ts.add_argument("--card-dir",
                    default="semantic_runs/latest/cards")
    ts.add_argument("--provider", default="openai",
                    choices=["stub", "anthropic", "openai", "xai", "grok", "mistral", "ollama", "peft-local"])
    ts.add_argument("--model", default="gpt-5.5")
    ts.add_argument("--max-card-chars", type=int, default=240)
    ts.add_argument("--max-techtrack-chars", type=int, default=45000)
    ts.add_argument("--max-tokens", type=int, default=9000)
    ts.add_argument("--out-dir", default="techtrack_synthesis_runs")
    ts.add_argument("--modules", nargs="*",
                    default=["M04", "M05", "M06", "M07", "M11", "M12"])
    ts.add_argument("--max-cards", type=int, default=24)
    ts.set_defaults(func=_cmd_techtrack_synthesis)

    pr = sub.add_parser("eval-rescore", help="Add quality metrics to existing eval JSON reports")
    pr.add_argument("reports", nargs="+", help="One or more eval JSON reports")
    pr.add_argument("--out", default=None, help="Comparison markdown path when multiple reports are given")
    pr.set_defaults(func=_cmd_eval_rescore)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

