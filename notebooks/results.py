import marimo

__generated_with = "0.22.5"
app = marimo.App(width="wide")


@app.cell
def _():
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo
    import polars as pl
    from src.analysis.figure_paths import (
        DIAGNOSTICS_SUMMARY_STEM,
        JUDGE_RELIABILITY_BY_SOURCE_STEM,
        JUDGE_RELIABILITY_RIDGE_STEM,
        PRIOR_PREDICTIVE_STEM,
        analysis_figure_paths,
    )
    from src.analysis.posterior_archive import load_posterior
    from src.analysis.posterior_queries import rank_judges
    from src.schemas import ExperimentConfig

    return (
        ExperimentConfig,
        DIAGNOSTICS_SUMMARY_STEM,
        JUDGE_RELIABILITY_BY_SOURCE_STEM,
        JUDGE_RELIABILITY_RIDGE_STEM,
        Path,
        PRIOR_PREDICTIVE_STEM,
        analysis_figure_paths,
        load_posterior,
        mo,
        pl,
        rank_judges,
        subprocess,
        sys,
    )


@app.cell
def _(ExperimentConfig, Path, mo):
    config_path = Path("configs/experiment.yaml")
    config = ExperimentConfig.from_yaml(config_path)
    refresh_button = mo.ui.run_button(
        label="Refresh analysis",
        kind="neutral",
        tooltip="Regenerate diagnostics and plot PNGs from current code and config.",
    )
    intro = mo.md(
        f"""
        # Bayesian LLM Judge Reliability

        This Marimo notebook is the presentation layer for the current experiment state.

        - Experiment: `{config.experiment.name}`
        - Model: `{config.model.type}`
        - Variant: `{config.model.variant}`
        - Judges configured: `{len(config.judges)}`
        - Item subset size: `{config.data.subset_size}`
        """
    )
    return config, config_path, intro, refresh_button


@app.cell
def _(config, config_path, mo, refresh_button, subprocess, sys):
    if not refresh_button.value:
        refresh_status = mo.md(
            "Use **Refresh analysis** to regenerate diagnostics and plot assets from the current code and config."
        )
    else:
        commands = [
            ("diagnostics", "src.analysis.diagnostics"),
            ("plots", "src.analysis.plots"),
        ]
        failures: list[str] = []
        for step_name, module_name in commands:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    module_name,
                    "--config",
                    str(config_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                stderr = (result.stderr or result.stdout or "").strip()
                failures.append(f"$ {step_name}\n{stderr}")
        if not failures:
            refresh_status = mo.md(
                f"Refreshed diagnostics and figures for `{config.experiment.name}` using `{config.model.variant}`."
            )
        else:
            failure_text = "\n\n".join(failures)
            refresh_status = mo.md(f"Analysis refresh failed.\n\n```text\n{failure_text}\n```")
    return refresh_status


@app.cell
def _(
    DIAGNOSTICS_SUMMARY_STEM,
    JUDGE_RELIABILITY_BY_SOURCE_STEM,
    JUDGE_RELIABILITY_RIDGE_STEM,
    PRIOR_PREDICTIVE_STEM,
    config,
    analysis_figure_paths,
    load_posterior,
    mo,
    pl,
    rank_judges,
):
    matrix_path = config.data.matrix_path
    posterior_path = config.inference.posterior_path
    figures_dir = config.figures_dir
    figure_paths = analysis_figure_paths(figures_dir)
    diagnostics_path = figure_paths[DIAGNOSTICS_SUMMARY_STEM]
    prior_path = figure_paths[PRIOR_PREDICTIVE_STEM]
    ridge_path = figure_paths[JUDGE_RELIABILITY_RIDGE_STEM]
    source_figure_path = figure_paths[JUDGE_RELIABILITY_BY_SOURCE_STEM]

    matrix = pl.read_parquet(matrix_path) if matrix_path.exists() else None
    posterior = load_posterior(posterior_path) if posterior_path.exists() else None
    ranking = rank_judges(posterior) if posterior is not None else None
    posterior_backend = str(posterior.get("backend", "unknown")) if posterior is not None else None
    source_figure_expected = posterior is not None and "theta_source" in posterior and "source_ids" in posterior
    ridge_status = "present" if ridge_path.exists() else ("missing" if posterior is not None else "not applicable")
    source_status = (
        "present" if source_figure_path.exists() else ("missing" if source_figure_expected else "not applicable")
    )

    status = mo.md(
        f"""
        ## Artifact Status

        - Matrix: `{"present" if matrix is not None else "missing"}`
        - Posterior: `{"present" if posterior is not None else "missing"}`
        - Posterior backend: `{posterior_backend or "n/a"}`
        - Diagnostics summary: `{"present" if diagnostics_path.exists() else "missing"}`
        - Prior predictive judge-means figure: `{"present" if prior_path.exists() else "missing"}`
        - Judge reliability figure: `{ridge_status}`
        - Source-aware figure: `{source_status}`
        """
    )
    return (
        matrix,
        diagnostics_path,
        posterior,
        prior_path,
        ranking,
        ridge_path,
        source_figure_path,
        status,
    )


@app.cell
def _(Path, mo):
    def image_panel(title: str, path: Path) -> object:
        return mo.vstack([mo.md(f"## {title}"), mo.image(path.read_bytes())])

    return (image_panel,)


@app.cell
def _(config_path, diagnostics_path, image_panel, mo):
    if diagnostics_path.exists():
        diagnostics_panel = image_panel("Diagnostics Summary", diagnostics_path)
    else:
        diagnostics_panel = mo.md(
            "## Diagnostics Summary\n"
            f"Run `uv run python -m src.analysis.diagnostics --config {config_path}` "
            "to create it."
        )
    return (diagnostics_panel,)


@app.cell
def _(config_path, image_panel, mo, prior_path):
    if prior_path.exists():
        prior_panel = image_panel("Prior Predictive Judge Means", prior_path)
    else:
        prior_panel = mo.md(
            "## Prior Predictive Judge Means\n"
            f"Run `uv run python -m src.analysis.plots --config {config_path}` "
            "to create it."
        )
    return (prior_panel,)


@app.cell
def _(config_path, image_panel, mo, ridge_path):
    if ridge_path.exists():
        ridge_panel = image_panel("Judge Reliability Figure", ridge_path)
    else:
        ridge_panel = mo.md(
            "## Judge Reliability Figure\n"
            f"Run `uv run python -m src.analysis.plots --config {config_path}` "
            "to create it."
        )
    return (ridge_panel,)


@app.cell
def _(image_panel, mo, posterior, source_figure_path):
    source_figure_expected = posterior is not None and "theta_source" in posterior and "source_ids" in posterior
    if source_figure_path.exists():
        source_panel = image_panel("Source-Aware Figure", source_figure_path)
    elif source_figure_expected:
        source_panel = mo.md("## Source-Aware Figure\nExpected for current posterior, but asset is missing.")
    else:
        source_panel = mo.md("## Source-Aware Figure\nNot generated for current posterior and config.")
    return (source_panel,)


@app.cell
def _(matrix, mo, ranking):
    outputs = [mo.md("## Results Overview")]
    if matrix is not None:
        outputs.append(mo.md(f"Processed matrix rows: `{matrix.height}`"))
        outputs.append(matrix.head(10))
    if ranking is not None:
        outputs.append(mo.md("### Posterior Judge Ranking"))
        outputs.append(ranking)
    if len(outputs) == 1:
        outputs.append(mo.md("No processed artifacts found yet. Run the pipeline first."))
    return mo.vstack(outputs)


@app.cell
def _(
    diagnostics_panel,
    intro,
    mo,
    prior_panel,
    refresh_button,
    refresh_status,
    ridge_panel,
    source_panel,
    status,
):
    dashboard = mo.vstack(
        [
            intro,
            mo.hstack([refresh_button], justify="start"),
            refresh_status,
            status,
            diagnostics_panel,
            prior_panel,
            ridge_panel,
            source_panel,
        ]
    )
    return (dashboard,)


@app.cell
def _(dashboard):
    dashboard  # noqa: B018


if __name__ == "__main__":
    app.run()
