# Data

This directory stores JudgeBench source snapshots, append-only judge logs, processed judge-by-item matrices, and posterior artifacts derived from them.

- `raw/`: optional cached JudgeBench exports or subset snapshots
- `logs/`: one JSONL log per judge configuration
- `processed/`: parquet matrices and posterior `.npz` outputs

JudgeBench is loaded from Hugging Face (`ScalerLab/JudgeBench`) under its published license. Keep large raw exports out of git when possible.

