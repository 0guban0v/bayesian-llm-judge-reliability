# Structure

```text
configs/experiment.yaml
  -> src/schemas.py
  -> src/data/loader.py
  -> src/judges/runner.py
  -> src/models/irt_numpyro.py
  -> src/analysis/*.py
  -> notebooks/results.py
```

- `configs/experiment.yaml`: single source of truth for data, judges, and inference
- `src/data/`: item preparation, matrix construction, validation
- `src/judges/`: prompts, parsing, MLX backend, runner
- `src/models/`: NumPyro and BlackJAX inference
- `src/analysis/`: diagnostics, figures, posterior queries
- `notebooks/results.py`: marimo presentation layer
