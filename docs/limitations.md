# Limitations

## Comparison Limits

- Model panel is not parameter-matched. DeepSeek 14B is directly compared against 7B and 9B models, so family effects and scale effects are not fully separated.
- Results are conditioned on one fixed pointwise protocol with constrained `FINAL VERDICT: A|B` decoding. They do not show how the same models behave under free-form judging, chain-of-thought prompting, or richer rubric-based evaluation.
- Current ranking is local to the present panel. Adding or removing models can change posterior separation and interpretation.

## Dataset Limits

- Results are based on a sampled subset of JudgeBench, not the full benchmark.
- Current sample mixes multiple source families; the active `source_hier` model captures judge-by-source effects, but source heterogeneity is still only partially modeled.
- Benchmark labels are treated as reference truth, but benchmark noise or ambiguity can still propagate into judge rankings.
- Removing category filtering broadens coverage, but also increases heterogeneity in task format and difficulty.

## Modeling Limits

- Current model learns one global reliability parameter per judge. It does not yet allow judge reliability to vary by source family, task type, or response model.
- The optional `source_hier` variant adds source-specific judge effects with partial pooling, but it still keeps item difficulty global and does not model richer source-by-item interactions.
- Healthy diagnostics do not rule out model misspecification. A well-mixed posterior can still be the wrong abstraction for the data-generating process.
- 2PL adds item discrimination flexibility, but that does not guarantee the latent structure is the best representation of LLM-as-judge behavior.

## Operational Limits

- Local MLX / Metal memory behavior constrains feasible model panels and execution order.
- Current judge backend is Apple Silicon / MLX oriented rather than a general cross-platform local inference path.
- Some candidate models are gated on Hugging Face or require manual local authentication before they can be included in the panel.
- Supported models must expose EOS token IDs and at least one single-token realization for both `A` and `B`; otherwise constrained verdict-only decoding is not valid.
- Performance metrics currently capture end-to-end runtime and peak RSS, but stage-level comparisons still require targeted profiling runs to isolate where improvements matter.

## Claim Limits

- Current evidence supports comparative reliability claims under this repo’s protocol, not broader claims about model quality in general.
- Posterior ranking should be read with uncertainty, not as an absolute leaderboard.
- Findings are strong enough to guide next experiments, but not strong enough to generalize across all judge prompts, all benchmarks, or all deployment conditions.
