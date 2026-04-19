"""Generate report-facing exports from current pipeline artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl

from src.analysis.diagnostics import diagnostic_parameter_rows, summarize_diagnostic_rows
from src.analysis.plot_config import judge_display_label
from src.analysis.posterior_archive import load_posterior
from src.analysis.posterior_queries import probability_judge_a_exceeds_b, rank_judges
from src.analysis.posterior_utils import (
    validate_posterior_plot_inputs,
)
from src.data.matrix_semantics import observed_accuracy_frame
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


class ModelComparisonUnavailableError(RuntimeError):
    """Expected absence/mismatch of study inputs for cross-run model comparison."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    return parser.parse_args()


def generated_dir(config: ExperimentConfig, output_dir: Path | None = None) -> Path:
    path = output_dir if output_dir is not None else config.report_dir / "generated"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tex_escape(value: object) -> str:
    return "".join(LATEX_ESCAPE_MAP.get(char, char) for char in str(value))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def compile_safe_note(message: str) -> str:
    return f"\\emph{{{tex_escape(message)}}}"


def project_root(config: ExperimentConfig) -> Path:
    """Return the repository root inferred from report/config paths."""

    return config.report_dir.parent


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


def write_results_exports(
    config: ExperimentConfig,
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray] | None,
    *,
    output_dir: Path | None = None,
) -> None:
    output_dir = generated_dir(config, output_dir=output_dir)
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
    *,
    output_dir: Path | None = None,
) -> None:
    output_dir = generated_dir(config, output_dir=output_dir)
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
        (f"{tex_escape(row['parameter'])} & {float(row['rhat_max']):.3f} & {float(row['ess_min']):.1f} {ROW_END}")
        for row in summary.to_dicts()
    ]
    write_text(
        output_dir / "diagnostics_summary.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\small",
                r"\begin{tabular}{lcc}",
                r"\toprule",
                f"parameter & max $\\hat{{R}}$ & min ESS {ROW_END}",
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


def _study_inferencedata_paths(config: ExperimentConfig) -> dict[str, dict[str, Path]]:
    """Return split-by-variant InferenceData paths from the explicit study config set."""

    config_dir = project_root(config) / "configs"
    expected_configs = {
        "gpt": {
            "global": config_dir / "experiment_gpt_global.yaml",
            "source_hier": config_dir / "experiment_gpt_source_hier.yaml",
        },
        "claude": {
            "global": config_dir / "experiment_claude_global.yaml",
            "source_hier": config_dir / "experiment_claude_source_hier.yaml",
        },
    }
    split_paths: dict[str, dict[str, Path]] = {}
    for split_name, variant_configs in expected_configs.items():
        split_paths[split_name] = {}
        for variant_name, config_path in variant_configs.items():
            if not config_path.exists():
                raise ModelComparisonUnavailableError(
                    f"Missing required study config for model comparison: {config_path}"
                )
            study_config = ExperimentConfig.from_yaml(config_path)
            if study_config.model.type != "2PL":
                raise ModelComparisonUnavailableError(f"Study config must use 2PL for model comparison: {config_path}")
            if study_config.model.variant != variant_name:
                raise ModelComparisonUnavailableError(
                    f"Study config variant mismatch for {config_path}: expected {variant_name}, "
                    f"found {study_config.model.variant}"
                )
            if study_config.data.splits != [split_name]:
                raise ModelComparisonUnavailableError(
                    f"Study config split mismatch for {config_path}: expected [{split_name}], "
                    f"found {study_config.data.splits}"
                )
            split_paths[split_name][variant_name] = study_config.inference.inferencedata_path
    return split_paths


def write_model_comparison_exports(
    config: ExperimentConfig,
    *,
    output_dir: Path | None = None,
) -> None:
    """Write a compact PSIS-LOO/WAIC comparison table for matched split-wise study runs."""

    output_dir = generated_dir(config, output_dir=output_dir)
    comparison_rows: list[str] = []
    missing_paths: list[Path] = []
    try:
        study_paths = _study_inferencedata_paths(config)
    except ModelComparisonUnavailableError as exc:
        write_text(output_dir / "model_comparison.tex", compile_safe_note(str(exc)))
        return
    for split_name, paths in study_paths.items():
        required_variants = {"global", "source_hier"}
        if set(paths) != required_variants:
            write_text(
                output_dir / "model_comparison.tex",
                compile_safe_note(
                    f"PSIS-LOO/WAIC comparison is unavailable because split {split_name} does not provide both "
                    "global and source_hier study outputs."
                ),
            )
            return
        if not all(path.exists() for path in paths.values()):
            missing_paths.extend(path for path in paths.values() if not path.exists())
            continue
        global_idata = az.from_netcdf(paths["global"])
        source_hier_idata = az.from_netcdf(paths["source_hier"])
        if not hasattr(global_idata, "log_likelihood") or not hasattr(source_hier_idata, "log_likelihood"):
            write_text(
                output_dir / "model_comparison.tex",
                compile_safe_note(
                    "PSIS-LOO/WAIC comparison is unavailable for current study archives. "
                    "Re-run inference after enabling saved pointwise log likelihood."
                ),
            )
            return
        loo_global = az.loo(global_idata)
        loo_source = az.loo(source_hier_idata)
        waic_global = az.waic(global_idata)
        waic_source = az.waic(source_hier_idata)
        compare = az.compare({"global": global_idata, "source_hier": source_hier_idata}, ic="loo", method="stacking")
        preferred = str(compare.index[0]).replace("_", "-")
        weight_source_hier = float(compare.loc["source_hier", "weight"])
        max_pareto_k = max(
            float(np.asarray(loo_global.pareto_k).max()),
            float(np.asarray(loo_source.pareto_k).max()),
        )
        comparison_rows.append(
            f"{tex_escape(split_name)} & "
            f"{float(loo_source.elpd_loo - loo_global.elpd_loo):.2f} & "
            f"{float(waic_source.elpd_waic - waic_global.elpd_waic):.2f} & "
            f"{tex_escape(preferred)} & "
            f"{weight_source_hier:.3f} & "
            f"{max_pareto_k:.3f} {ROW_END}"
        )

    if missing_paths:
        write_text(
            output_dir / "model_comparison.tex",
            compile_safe_note(
                "PSIS-LOO/WAIC comparison is unavailable because one or more split-study InferenceData files are "
                "missing."
            ),
        )
        return

    write_text(
        output_dir / "model_comparison.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\small",
                r"\begin{tabular}{lccccc}",
                r"\toprule",
                (
                    "split & $\\Delta$ elpd$_{loo}$ & $\\Delta$ elpd$_{waic}$ & "
                    f"LOO winner & stacking wt. source-hier & max Pareto $k$ {ROW_END}"
                ),
                r"\midrule",
                *comparison_rows,
                r"\bottomrule",
                r"\end{tabular}",
                (
                    r"\caption{Matched split-wise model comparison between global and source-hierarchical 2PL fits. "
                    r"Positive $\Delta$ values favor source-hier over global.}"
                ),
                r"\label{tab:model-comparison}",
                r"\end{table}",
            ]
        ),
    )


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    config.ensure_directories()
    matrix = pl.read_parquet(config.data.matrix_path) if config.data.matrix_path.exists() else None
    posterior = load_posterior(config.inference.posterior_path) if config.inference.posterior_path.exists() else None

    if matrix is not None and posterior is not None:
        validate_posterior_plot_inputs(matrix, posterior)

    if matrix is not None:
        write_results_exports(config, matrix, posterior)
    else:
        output_dir = generated_dir(config)
        write_text(output_dir / "judge_summary.tex", compile_safe_note("Judge matrix missing for this run."))
        write_text(output_dir / "pairwise_summary.tex", compile_safe_note("Judge matrix missing for this run."))

    write_diagnostics_exports(config, posterior)
    write_model_comparison_exports(config)


if __name__ == "__main__":
    main()
