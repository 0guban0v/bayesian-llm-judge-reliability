"""Generate report-facing exports from current pipeline artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import polars as pl

from src.analysis.diagnostics import diagnostic_parameter_rows, summarize_diagnostic_rows
from src.analysis.plot_config import judge_display_label, source_display_label
from src.analysis.posterior_archive import load_posterior
from src.analysis.posterior_queries import probability_judge_a_exceeds_b, rank_judges
from src.analysis.posterior_utils import (
    has_source_reliability,
    source_reliability_summary,
    top_source_ids,
    validate_posterior_plot_inputs,
)
from src.data.loader import load_judge_logs
from src.data.matrix_semantics import first_original_judgments, observed_accuracy_frame, pivot_original_judgments
from src.schemas import ExperimentConfig

ROW_END = r"\\"
LATEX_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    return parser.parse_args()


def generated_dir(config: ExperimentConfig) -> Path:
    path = config.report_dir / "generated"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tex_escape(value: object) -> str:
    return "".join(LATEX_ESCAPE_MAP.get(char, char) for char in str(value))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def compile_safe_note(message: str) -> str:
    return f"\\emph{{{tex_escape(message)}}}"


def observed_accuracy(matrix: pl.DataFrame, judge_ids: list[str]) -> dict[str, float]:
    observed = observed_accuracy_frame(matrix, judge_ids)
    return dict(zip(observed.get_column("judge_id").to_list(), observed.get_column("accuracy").to_list(), strict=True))


def representative_pairwise_rows(
    posterior: dict[str, np.ndarray],
    ranking: pl.DataFrame,
) -> list[tuple[str, str, float]]:
    ordered_judges = ranking.get_column("judge_id").cast(pl.String).to_list()
    if len(ordered_judges) < 2:
        return []

    candidate_indices = [(0, 1), (0, 2), (1, 2), (-3, -1), (-2, -1)]
    selected_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for left_idx, right_idx in candidate_indices:
        try:
            judge_a = ordered_judges[left_idx]
            judge_b = ordered_judges[right_idx]
        except IndexError:
            continue
        if judge_a == judge_b or (judge_a, judge_b) in seen:
            continue
        seen.add((judge_a, judge_b))
        selected_pairs.append((judge_a, judge_b))

    if not selected_pairs:
        selected_pairs.append((ordered_judges[0], ordered_judges[1]))

    return [
        (
            judge_display_label(judge_a),
            judge_display_label(judge_b),
            probability_judge_a_exceeds_b(posterior, judge_a, judge_b),
        )
        for judge_a, judge_b in selected_pairs
    ]


def selected_focus_judge(judge_ids: list[str], preferred_judge_id: str | None) -> str:
    if preferred_judge_id and preferred_judge_id in judge_ids:
        return preferred_judge_id
    return judge_ids[0]


def standout_cases(
    items: pl.DataFrame,
    logs: pl.DataFrame,
    judge_ids: list[str],
    *,
    preferred_judge_id: str | None,
    limit: int,
) -> tuple[str, pl.DataFrame]:
    focus_judge = selected_focus_judge(judge_ids, preferred_judge_id)
    if logs.height == 0:
        return focus_judge, pl.DataFrame([])

    original_logs = logs.filter(pl.col("prompt_order").eq("original") & pl.col("correct").is_not_null())
    if original_logs.height == 0:
        return focus_judge, pl.DataFrame([])

    first_correctness = first_original_judgments(logs)
    if first_correctness.height == 0:
        return focus_judge, pl.DataFrame([])
    pivoted = pivot_original_judgments(first_correctness)
    other_judges = [judge_id for judge_id in judge_ids if judge_id != focus_judge and judge_id in pivoted.columns]
    filtered = pivoted.filter(
        pl.col(focus_judge).fill_null(0).eq(1)
        & pl.all_horizontal([pl.col(judge_id).fill_null(0).eq(0) for judge_id in other_judges])
    )
    if filtered.height == 0:
        return focus_judge, pl.DataFrame([])

    focus_logs = (
        original_logs.filter(pl.col("judge_id").eq(focus_judge))
        .group_by("item_id", maintain_order=True)
        .agg(
            [
                pl.col("raw_response").first().alias("raw_response"),
                pl.col("latency_ms").first().alias("latency_ms"),
            ]
        )
    )
    available_judge_ids = [j for j in judge_ids if j in filtered.columns]
    selected = (
        items.join(filtered.select(["item_id", *available_judge_ids]), on="item_id", how="inner")
        .join(focus_logs, on="item_id", how="left")
        .sort("item_id")
        .select(
            [
                "item_id",
                "source",
                "label",
                "response_a",
                "response_b",
                "raw_response",
                *available_judge_ids,
            ]
        )
        .head(limit)
    )
    return focus_judge, selected


def write_results_exports(
    config: ExperimentConfig,
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray] | None,
) -> None:
    output_dir = generated_dir(config)
    judge_ids = [judge.id for judge in config.judges]
    accuracy_by_judge = observed_accuracy(matrix, judge_ids)
    if posterior is None:
        write_text(output_dir / "judge_summary.tex", compile_safe_note("Posterior artifact missing for this run."))
        write_text(output_dir / "pairwise_summary.tex", compile_safe_note("Posterior artifact missing for this run."))
        return

    ranking = rank_judges(posterior).with_columns(pl.col("judge_id").cast(pl.String))
    accuracy_frame = pl.DataFrame(
        {"judge_id": list(accuracy_by_judge.keys()), "accuracy": list(accuracy_by_judge.values())}
    )
    ranking = ranking.join(accuracy_frame, on="judge_id", how="left")
    if ranking.get_column("accuracy").null_count() > 0:
        missing = ", ".join(ranking.filter(pl.col("accuracy").is_null()).get_column("judge_id").to_list())
        raise ValueError(f"Missing observed accuracy for judges: {missing}")

    judge_rows = [
        (
            f"{tex_escape(judge_display_label(str(row['judge_id'])))} & {float(row['accuracy']):.3f} & "
            f"{float(row['theta_mean']):.3f} & [{float(row['theta_p05']):.3f}, {float(row['theta_p95']):.3f}] {ROW_END}"
        )
        for row in ranking.to_dicts()
    ]
    write_text(
        output_dir / "judge_summary.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\small",
                r"\begin{tabular}{lccc}",
                r"\toprule",
                f"judge & accuracy & posterior $\\theta$ mean & 90\\% CI {ROW_END}",
                r"\midrule",
                *judge_rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Observed accuracy and posterior judge reliability for current run.}",
                r"\label{tab:judge-summary}",
                r"\end{table}",
            ]
        ),
    )

    pairwise_lines = [
        f"{tex_escape(left_label)} $>$ {tex_escape(right_label)} & {probability:.3f} {ROW_END}"
        for left_label, right_label, probability in representative_pairwise_rows(posterior, ranking)
    ]
    write_text(
        output_dir / "pairwise_summary.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\small",
                r"\begin{tabular}{lc}",
                r"\toprule",
                f"comparison & $P(\\theta_a > \\theta_b \\mid y)$ {ROW_END}",
                r"\midrule",
                *pairwise_lines,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Representative pairwise posterior comparison probabilities.}",
                r"\label{tab:pairwise}",
                r"\end{table}",
            ]
        ),
    )


def write_diagnostics_exports(
    config: ExperimentConfig,
    posterior: dict[str, np.ndarray] | None,
) -> None:
    output_dir = generated_dir(config)
    if posterior is None:
        write_text(
            output_dir / "diagnostics_summary.tex",
            compile_safe_note("Posterior artifact missing for this run."),
        )
        return
    rows = diagnostic_parameter_rows(posterior)
    divergences = int(np.asarray(posterior.get("diverging", np.array([]))).sum())
    summary = summarize_diagnostic_rows(rows, divergences)
    table_lines = [
        (
            f"{tex_escape(row['parameter'])} & {float(row['rhat_max']):.3f} & "
            f"{float(row['ess_min']):.1f} & {int(row['divergences'])} {ROW_END}"
        )
        for row in summary.to_dicts()
    ]
    write_text(
        output_dir / "diagnostics_summary.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\small",
                r"\begin{tabular}{lccc}",
                r"\toprule",
                f"parameter & max $\\hat{{R}}$ & min ESS & divergences {ROW_END}",
                r"\midrule",
                *table_lines,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Compact posterior diagnostics exported from current run.}",
                r"\label{tab:diagnostics}",
                r"\end{table}",
            ]
        ),
    )


def write_source_exports(
    config: ExperimentConfig,
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray] | None,
) -> None:
    output_dir = generated_dir(config)
    if posterior is None or not has_source_reliability(posterior):
        write_text(
            output_dir / "source_summary.tex",
            compile_safe_note("Source-aware posterior summary not available for this run."),
        )
        return

    ordered_sources = top_source_ids(matrix, posterior, max_sources=config.analysis.plots.max_sources)
    summary = source_reliability_summary(posterior, ordered_sources)
    lines = []
    for source_id in ordered_sources:
        source_rows = summary.filter(pl.col("source") == source_id)
        best_row = max(source_rows.to_dicts(), key=lambda row: float(row["theta_mean"]))
        source_label = tex_escape(source_display_label(source_id))
        best_judge_label = tex_escape(judge_display_label(str(best_row["judge_id"])))
        interval = f"[{float(best_row['theta_p05']):.3f}, {float(best_row['theta_p95']):.3f}]"
        lines.append(
            f"{source_label} & {best_judge_label} & {float(best_row['theta_mean']):.3f} & {interval} {ROW_END}"
        )
    write_text(
        output_dir / "source_summary.tex",
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\small",
                r"\centering",
                r"\begin{tabular}{lccc}",
                r"\toprule",
                f"source & best judge & mean $\\theta_{{j,s}}$ & 90\\% interval {ROW_END}",
                r"\midrule",
                *lines,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Top plotted sources and strongest judge by posterior mean source-specific reliability.}",
                r"\label{tab:source-summary}",
                r"\end{table}",
            ]
        ),
    )


def _response_synopsis(text: str, max_chars: int) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    snippet = sentences[-1] if sentences else text.strip()
    return snippet[:max_chars] + ("\u2026" if len(snippet) > max_chars else "")


def write_case_exports(config: ExperimentConfig, items: pl.DataFrame | None) -> None:
    output_dir = generated_dir(config)
    if items is None:
        write_text(output_dir / "standout_cases.tex", compile_safe_note("Item subset is missing for this run."))
        return

    judge_ids = [judge.id for judge in config.judges]
    focus_judge = selected_focus_judge(judge_ids, config.analysis.report.standout_judge_id)
    logs = load_judge_logs(config.data.logs_dir)
    focus_judge, cases = standout_cases(
        items,
        logs,
        judge_ids,
        preferred_judge_id=config.analysis.report.standout_judge_id,
        limit=config.analysis.report.standout_case_limit,
    )
    focus_label = judge_display_label(focus_judge)
    if cases.height == 0:
        write_text(
            output_dir / "standout_cases.tex",
            compile_safe_note(f"No original-order cases matched the selection rule for {focus_label}."),
        )
        return

    table_rows = []
    for row in cases.to_dicts():
        verdict_line = str(row.get("raw_response") or "").splitlines()[0].strip()
        table_rows.append(
            " & ".join(
                [
                    tex_escape(source_display_label(str(row.get("source") or "unknown"))),
                    tex_escape(row.get("label") or "n/a"),
                    tex_escape(
                        _response_synopsis(
                            str(row.get("response_a") or ""),
                            config.analysis.report.response_synopsis_chars,
                        )
                    ),
                    tex_escape(
                        _response_synopsis(
                            str(row.get("response_b") or ""),
                            config.analysis.report.response_synopsis_chars,
                        )
                    ),
                    tex_escape(verdict_line or "n/a"),
                ]
            )
            + f" {ROW_END}"
        )

    write_text(
        output_dir / "standout_cases.tex",
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\captionsetup{justification=raggedright,singlelinecheck=false}",
                r"\footnotesize",
                r"\setlength{\tabcolsep}{3pt}",
                (
                    r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.12\textwidth}"
                    r">{\raggedright\arraybackslash}p{0.06\textwidth}"
                    r">{\raggedright\arraybackslash}p{0.33\textwidth}"
                    r">{\raggedright\arraybackslash}p{0.33\textwidth}"
                    r">{\raggedright\arraybackslash}p{0.11\textwidth}@{}}"
                ),
                r"\toprule",
                (
                    f"source & label & resp A (conclusion) & resp B (conclusion) & "
                    f"{tex_escape(focus_label)} verdict {ROW_END}"
                ),
                r"\midrule",
                *table_rows,
                r"\bottomrule",
                r"\end{tabular}",
                (
                    f"\\caption{{Readable standout cases where only {tex_escape(focus_label)} "
                    "is correct and every other configured judge is incorrect under original prompt order.}"
                ),
                r"\label{tab:standout-cases}",
                r"\end{table}",
            ]
        ),
    )


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    config.ensure_directories()
    items = pl.read_parquet(config.data.item_path) if config.data.item_path.exists() else None
    matrix = pl.read_parquet(config.data.matrix_path) if config.data.matrix_path.exists() else None
    posterior = load_posterior(config.inference.posterior_path) if config.inference.posterior_path.exists() else None

    if matrix is not None and posterior is not None:
        validate_posterior_plot_inputs(matrix, posterior)

    if matrix is not None:
        write_results_exports(config, matrix, posterior)
        write_source_exports(config, matrix, posterior)
    else:
        output_dir = generated_dir(config)
        write_text(output_dir / "judge_summary.tex", compile_safe_note("Judge matrix missing for this run."))
        write_text(output_dir / "pairwise_summary.tex", compile_safe_note("Judge matrix missing for this run."))
        write_text(output_dir / "source_summary.tex", compile_safe_note("Judge matrix missing for this run."))

    write_diagnostics_exports(config, posterior)
    write_case_exports(config, items)


if __name__ == "__main__":
    main()
