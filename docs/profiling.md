# Profiling

Use `py-spy` for low-overhead Python profiling of the main pipeline stages.

Install profiling dependencies first:

```bash
uv sync
```

If you need a standalone install instead:

```bash
brew install py-spy
```

or

```bash
cargo install py-spy
```

`py-spy` is an external sampling profiler written in Rust. It can record a Python command directly and export a flamegraph or speedscope profile. Source: [py-spy on PyPI](https://pypi.org/project/py-spy/0.4.0/).

## Targets

```bash
make profile-judge JUDGE=deepseek-r1-distill-qwen-14b LIMIT=10
make profile-matrix
make profile-infer
make profile-plots
make profile-full
```

Each target writes:

- a speedscope profile to `profiles/`
- a summary JSON to `profiles/metrics/`

The summary JSON includes wall-clock time, peak RSS, target name, config path, profile path, and exit code, so before/after comparisons do not require opening the trace first.

## Notes

- `profile-judge` is the highest-signal target for this repo because local judge inference dominates runtime.
- `profile-matrix` is useful if JSONL loading or matrix rebuild starts to matter at larger subset sizes.
- `profile-infer` helps if posterior sampling becomes the bottleneck after scaling items or judges.
- `profile-full` records full-run wall-clock and peak-RSS metrics in `profiles/metrics/`. It does not emit a `py-spy` trace for the whole Make process tree.
- `py-spy` is for CPU time profiling, not memory-leak analysis.
