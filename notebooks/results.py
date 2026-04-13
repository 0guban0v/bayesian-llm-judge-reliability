import marimo

__generated_with = "0.11.0"
app = marimo.App(width="wide")


@app.cell
def __():
    from pathlib import Path

    import marimo as mo
    import polars as pl
    from src.analysis.diagnostics import load_posterior
    from src.analysis.posterior_queries import rank_judges
    from src.schemas import ExperimentConfig

    return ExperimentConfig, Path, load_posterior, mo, pl, rank_judges


@app.cell
def __(ExperimentConfig, mo):
    config = ExperimentConfig.from_yaml("configs/experiment.yaml")
    intro = mo.md(
        f"""
        # Bayesian LLM Judge Reliability

        This notebook is the presentation layer for the project.

        - Experiment: `{config.experiment.name}`
        - Model: `{config.model.type}`
        - Judges configured: `{len(config.judges)}`
        - Item subset size: `{config.data.subset_size}`
        """
    )
    return config, intro


@app.cell
def __(intro, mo):
    return mo.vstack([intro])


@app.cell
def __(Path, config, load_posterior, mo, pl, rank_judges):
    matrix_path = config.data.matrix_path
    posterior_path = config.inference.posterior_path
    hero_path = Path("figures/judge_reliability_ridge.png")
    source_figure_path = Path("figures/judge_reliability_by_source.png")

    matrix = pl.read_parquet(matrix_path) if matrix_path.exists() else None
    posterior = load_posterior(posterior_path) if posterior_path.exists() else None
    ranking = rank_judges(posterior) if posterior is not None else None

    status = mo.md(
        f"""
        ## Artifact Status

        - Matrix: `{"present" if matrix is not None else "missing"}`
        - Posterior: `{"present" if posterior is not None else "missing"}`
        - Hero figure: `{"present" if hero_path.exists() else "missing"}`
        - Source-aware figure: `{"present" if source_figure_path.exists() else "missing"}`
        """
    )
    return hero_path, matrix, posterior, ranking, source_figure_path, status


@app.cell
def __(status, mo):
    return mo.vstack([status])


@app.cell
def __(hero_path, mo):
    if hero_path.exists():
        panel = mo.vstack([mo.md("## Hero Figure"), mo.image(hero_path)])
    else:
        panel = mo.md(
            "## Hero Figure\n"
            "Run `uv run python -m src.analysis.plots --config configs/experiment.yaml` "
            "to create it."
        )
    return panel


@app.cell
def __(mo, source_figure_path):
    if source_figure_path.exists():
        panel = mo.vstack([mo.md("## Source-Aware Figure"), mo.image(source_figure_path)])
    else:
        panel = mo.md("## Source-Aware Figure\nNo source-aware figure generated for this run.")
    return panel


@app.cell
def __(matrix, mo, ranking):
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


if __name__ == "__main__":
    app.run()
