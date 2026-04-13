# Archive Index

This folder preserves earlier experiment states that are no longer the active pipeline outputs.

## Folder map

- `data/logs/prompt-variant-ablation/`
- `data/processed/prompt-variant-ablation/`
- `figures/prompt-variant-ablation/`
  - 200-item DeepSeek-R1-Distill-Qwen-14B prompt-variant ablation.
  - Includes `pointwise`, `pointwise-cot`, `pairwise`, and `pairwise-cot` judge columns.
  - Archived processed matrix shows all four columns are fully observed, have accuracy `0.625`, and match item by item.

- `data/logs/three-model-panel/`
- `data/processed/three-model-panel/`
- `figures/three-model-panel/`
  - 200-item early multi-model panel.
  - Processed matrix contains DeepSeek-R1-Distill-Qwen-14B, Mistral-7B-Instruct-v0.3, and Qwen2.5-7B-Instruct.
  - The logs also contain a short `qwen2-5-14b-instruct.jsonl` smoke run that did not enter the processed matrix.

- `data/logs/four-model-panel-500item/`
- `data/processed/four-model-panel-500item/`
- `figures/four-model-panel-500item/`
  - 500-item later multi-model panel with complete coverage.
  - Processed matrix contains DeepSeek-R1-Distill-Qwen-14B, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, and Gemma-2-9B-IT.
  - This archive is not the same as the current configured five-model experiment, which additionally includes DeepSeek-R1-Distill-Qwen-7B.
