# Assumptions

## Experimental Assumptions

- JudgeBench preference labels are treated as the reference outcome for judge evaluation.
- The current `gpt` and `claude` split sample is informative enough to compare judge models on the chosen protocol.
- A fixed constrained verdict-only judging protocol is a valid way to compare model reliability without confounding on formatting compliance.
- Current local MLX generation path is operationally stable enough that observed verdicts reflect model behavior rather than parser drift or prompt-format failures.

## Modeling Assumptions

- IRT is an appropriate latent-variable model for this setting: judges differ in reliability, items differ in difficulty, and item discrimination can vary under 2PL.
- Judges are exchangeable units under the prior before observing results, even though they come from different model families and parameter scales.
- Items are exchangeable under the prior after conditioning on inclusion in the sampled benchmark subset.
- Posterior judge ranking is meaningful only relative to the current protocol, benchmark subset, and model panel.
- Binary correctness derived from pairwise labels is an adequate observation model for downstream Bayesian inference.

## Reproducibility Assumptions

- `experiment.seed` is the root source of stochasticity for sampling and inference configuration.
- Config file is the single source of truth for model panel, dataset selection, and inference settings.
- Append-only JSONL judge logs and derived parquet / posterior artifacts are sufficient for audit and rerun provenance.

## Interpretation Assumptions

- Higher posterior `theta` corresponds to more reliable agreement with benchmark labels under the current judge protocol.
- Posterior predictive checks and standard MCMC diagnostics are necessary but not sufficient for scientific validity; they assess fit and sampler behavior, not benchmark truth.
- Source metadata preserved in the dataset is meaningful and can support later stratified analyses without changing the underlying benchmark items.
