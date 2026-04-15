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
    from src.analysis.diagnostics import load_posterior
    from src.analysis.figure_paths import (
        JUDGE_RELIABILITY_BY_SOURCE_STEM,
        JUDGE_RELIABILITY_RIDGE_STEM,
        PRIOR_PREDICTIVE_STEM,
        figure_base_path,
    )
    from src.analysis.posterior_queries import rank_judges
    from src.schemas import ExperimentConfig

    return (
        ExperimentConfig,
        JUDGE_RELIABILITY_BY_SOURCE_STEM,
        JUDGE_RELIABILITY_RIDGE_STEM,
        Path,
        PRIOR_PREDICTIVE_STEM,
        figure_base_path,
        load_posterior,
        mo,
        pl,
        rank_judges,
        subprocess,
        sys,
    )


@app.cell
def _(ExperimentConfig, mo):
    config = ExperimentConfig.from_yaml("configs/experiment.yaml")
    refresh_button = mo.ui.run_button(
        label="Refresh figures",
        kind="neutral",
        tooltip="Regenerate plot PNGs from current code and config.",
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
    return config, intro, refresh_button


@app.cell
def _(config, mo, refresh_button, subprocess, sys):
    if not refresh_button.value:
        refresh_status = mo.md(
            "Use **Refresh figures** to regenerate plot assets from the current code and config."
        )
    else:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.analysis.plots",
                "--config",
                "configs/experiment.yaml",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            refresh_status = mo.md(
                f"Refreshed figures for `{config.experiment.name}` using `{config.model.variant}`."
            )
        else:
            stderr = (result.stderr or result.stdout or "").strip()
            refresh_status = mo.md(f"Figure refresh failed.\n\n```text\n{stderr}\n```")
    return refresh_status


@app.cell
def _(
    JUDGE_RELIABILITY_BY_SOURCE_STEM,
    JUDGE_RELIABILITY_RIDGE_STEM,
    PRIOR_PREDICTIVE_STEM,
    config,
    figure_base_path,
    load_posterior,
    mo,
    pl,
    rank_judges,
):
    matrix_path = config.data.matrix_path
    posterior_path = config.inference.posterior_path
    figures_dir = config.figures_dir
    prior_path = figure_base_path(figures_dir, PRIOR_PREDICTIVE_STEM).with_suffix(".png")
    ridge_path = figure_base_path(figures_dir, JUDGE_RELIABILITY_RIDGE_STEM).with_suffix(".png")
    source_figure_path = figure_base_path(
        figures_dir,
        JUDGE_RELIABILITY_BY_SOURCE_STEM,
    ).with_suffix(".png")

    matrix = pl.read_parquet(matrix_path) if matrix_path.exists() else None
    posterior = load_posterior(posterior_path) if posterior_path.exists() else None
    ranking = rank_judges(posterior) if posterior is not None else None
    posterior_backend = str(posterior.get("backend", "unknown")) if posterior is not None else None

    status = mo.md(
        f"""
        ## Artifact Status

        - Matrix: `{"present" if matrix is not None else "missing"}`
        - Posterior: `{"present" if posterior is not None else "missing"}`
        - Posterior backend: `{posterior_backend or "n/a"}`
        - Prior predictive figure: `{"present" if prior_path.exists() else "missing"}`
        - Judge reliability figure: `{"present" if ridge_path.exists() else "missing"}`
        - Source-aware figure: `{"present" if source_figure_path.exists() else "missing"}`
        """
    )
    return (
        matrix,
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
def _(image_panel, mo, prior_path):
    if prior_path.exists():
        prior_panel = image_panel("Prior Predictive Figure", prior_path)
    else:
        prior_panel = mo.md(
            "## Prior Predictive Figure\n"
            "Run `uv run python -m src.analysis.plots --config configs/experiment.yaml` "
            "to create it."
        )
    return (prior_panel,)


@app.cell
def _(image_panel, mo, ridge_path):
    if ridge_path.exists():
        ridge_panel = image_panel("Judge Reliability Figure", ridge_path)
    else:
        ridge_panel = mo.md(
            "## Judge Reliability Figure\n"
            "Run `uv run python -m src.analysis.plots --config configs/experiment.yaml` "
            "to create it."
        )
    return (ridge_panel,)


@app.cell
def _(image_panel, mo, source_figure_path):
    if source_figure_path.exists():
        source_panel = image_panel("Source-Aware Figure", source_figure_path)
    else:
        source_panel = mo.md(
            "## Source-Aware Figure\nNo source-aware figure generated for this run."
        )
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
