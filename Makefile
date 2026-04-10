SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:

UV ?= uv
CONFIG ?= configs/experiment.yaml
JUDGE ?=
LIMIT ?=
LOG_DIR ?= logs
UV_CACHE_DIR ?= .uv-cache

export UV_PROJECT_ENVIRONMENT := .venv
export UV_CACHE_DIR

.PHONY: sync lint format dead-code quality test pre-commit-install pre-commit-run smoke recommend-models verify-models setup-models items items-refresh judge matrix validate infer infer-blackjax diagnostics plots notebook run

define log_path
$(LOG_DIR)/$(1)_$(shell date +'%Y%m%d_%H%M%S').log
endef

define run_and_log
	mkdir -p $(LOG_DIR)
	if [ -f .env ]; then set -a; . ./.env; set +a; fi
	echo "logging to $(call log_path,$(1))"
	$(2) 2>&1 | tee >(perl -ne 'BEGIN { $$| = 1 } s/\r/\n/g; print' > "$(call log_path,$(1))")
endef

sync:
	@$(UV) sync

lint:
	@$(UV) run ruff check src tests notebooks --fix

format:
	@$(UV) run ruff format src tests notebooks

dead-code:
	@$(UV) run vulture src tests tests/vulture_whitelist.py

quality:
	@$(MAKE) lint
	@$(MAKE) format
	@$(MAKE) dead-code

test:
	@MPLCONFIGDIR=.uv-cache/matplotlib $(UV) run python -m unittest discover -s tests

pre-commit-install:
	@$(UV) run pre-commit install --hook-type pre-commit --hook-type pre-push

pre-commit-run:
	@MPLCONFIGDIR=.uv-cache/matplotlib $(UV) run pre-commit run --all-files

smoke: SMOKE_JUDGE_LIMIT = 5
smoke: SMOKE_JUDGE_1 = deepseek-r1-distill-qwen-14b
smoke: SMOKE_JUDGE_2 = qwen2-5-7b-instruct
smoke: SMOKE_JUDGE_3 = mistral-7b-instruct-v0-3
smoke:
	@$(MAKE) judge CONFIG=$(CONFIG) JUDGE=$(SMOKE_JUDGE_1) LIMIT=$(SMOKE_JUDGE_LIMIT)
	@$(MAKE) judge CONFIG=$(CONFIG) JUDGE=$(SMOKE_JUDGE_2) LIMIT=$(SMOKE_JUDGE_LIMIT)
	@$(MAKE) judge CONFIG=$(CONFIG) JUDGE=$(SMOKE_JUDGE_3) LIMIT=$(SMOKE_JUDGE_LIMIT)
	@$(MAKE) judge CONFIG=$(CONFIG)
	@$(MAKE) matrix CONFIG=$(CONFIG)
	@$(MAKE) validate CONFIG=$(CONFIG)
	@$(MAKE) infer CONFIG=$(CONFIG)
	@$(MAKE) diagnostics CONFIG=$(CONFIG)
	@$(MAKE) plots CONFIG=$(CONFIG)

recommend-models:
	@$(call run_and_log,recommend_models,bash scripts/recommend_models.sh)

verify-models:
	@$(call run_and_log,verify_models,$(UV) run python scripts/verify_models.py $(MODELS))

setup-models:
	@$(call run_and_log,setup_models,bash scripts/setup_models.sh $(CONFIG))

items:
	@$(call run_and_log,items,$(UV) run python -m src.data.loader --config $(CONFIG))

items-refresh:
	@$(call run_and_log,items_refresh,$(UV) run python -m src.data.loader --config $(CONFIG) --refresh-items)

judge:
	@$(call run_and_log,judge,$(UV) run python -m src.judges.runner --config $(CONFIG) $(if $(JUDGE),--judge $(JUDGE),) $(if $(LIMIT),--limit $(LIMIT),))

matrix:
	@$(call run_and_log,matrix,$(UV) run python -m src.data.loader --config $(CONFIG) --rebuild-matrix)

validate:
	@$(call run_and_log,validate,$(UV) run python -m src.data.validate --config $(CONFIG))

infer:
	@$(call run_and_log,infer,bash scripts/infer.sh $(CONFIG))

infer-blackjax:
	@$(call run_and_log,infer_blackjax,$(UV) run python -m src.models.irt_blackjax --config $(CONFIG))

diagnostics:
	@$(call run_and_log,diagnostics,$(UV) run python -m src.analysis.diagnostics --config $(CONFIG))

plots:
	@$(call run_and_log,plots,$(UV) run python -m src.analysis.plots --config $(CONFIG))

notebook:
	@$(call run_and_log,notebook,$(UV) run marimo edit notebooks/results.py)

run:
	@$(call run_and_log,run,./scripts/run_experiment.sh $(CONFIG))
