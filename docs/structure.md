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
  Why: generated artifacts are derived from a reproducible sampled item set rather than ad hoc runs.
- `src/judges/`: prompts, parsing, MLX backend, runner
  Why: judge behavior is defined by the whole harness, not just the model ID.
- `src/models/`: NumPyro and BlackJAX inference
- `src/analysis/`: diagnostics, figures, posterior queries
- `notebooks/results.py`: marimo presentation layer

## Critical Building Blocks

- `data/logs/*.jsonl` are the canonical run records.
  What: append-only judge outputs with item metadata and parsed verdicts.
  Why: matrix, validation, and posterior artifacts can be rebuilt from logs without trusting in-memory run state.

- `src/judges/mlx_backend.py` implements constrained verdict-only decoding.
  What: assistant-side prefill plus a logits processor that allows only a verdict token and EOS.
  Why: prompt instructions alone were not enough to keep local judges format-stable.

- `src/judges/runner.py` executes judges sequentially and clears MLX model cache between judges.
  What: one-process orchestration with explicit cache cleanup after each judge.
  Why: local MLX / Metal memory behavior made multi-model lifecycle management part of the architecture.

- Judge execution is resumable at the log layer.
  What: runner skips already logged `(item_id, prompt_order)` pairs when appending to a judge JSONL.
  Why: interrupted judge runs can continue safely, but downstream artifacts are only meaningful once intended judge coverage is complete.
