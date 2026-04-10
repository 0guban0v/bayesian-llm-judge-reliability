.PHONY: smoke recommend-models verify-models setup-models items items-refresh judge matrix validate infer infer-blackjax diagnostics plots notebook full

smoke: SMOKE_JUDGE_LIMIT = 5
smoke: SMOKE_JUDGE_1 = deepseek-r1-distill-qwen-14b
smoke: SMOKE_JUDGE_2 = qwen2-5-7b-instruct
smoke: SMOKE_JUDGE_3 = mistral-7b-instruct-v0-3
smoke: SMOKE_JUDGE_4 = gemma-2-9b-it
smoke:
	@$(MAKE) judge CONFIG=$(CONFIG) JUDGE=$(SMOKE_JUDGE_1) LIMIT=$(SMOKE_JUDGE_LIMIT)
	@$(MAKE) judge CONFIG=$(CONFIG) JUDGE=$(SMOKE_JUDGE_2) LIMIT=$(SMOKE_JUDGE_LIMIT)
	@$(MAKE) judge CONFIG=$(CONFIG) JUDGE=$(SMOKE_JUDGE_3) LIMIT=$(SMOKE_JUDGE_LIMIT)
	@$(MAKE) judge CONFIG=$(CONFIG) JUDGE=$(SMOKE_JUDGE_4) LIMIT=$(SMOKE_JUDGE_LIMIT)
	@$(MAKE) judge CONFIG=$(CONFIG)
	@$(MAKE) matrix CONFIG=$(CONFIG)
	@$(MAKE) validate CONFIG=$(CONFIG)
	@$(MAKE) infer CONFIG=$(CONFIG)
	@$(MAKE) diagnostics CONFIG=$(CONFIG)
	@$(MAKE) plots CONFIG=$(CONFIG)

recommend-models:
	@$(call run_and_log,recommend_models,bash scripts/recommend_models.sh)

verify-models:
	@if [ -z "$(strip $(MODELS))" ]; then \
		echo 'MODELS is required. Example: make verify-models MODELS="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B Qwen/Qwen2.5-7B-Instruct"'; \
		exit 1; \
	fi
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
	@$(call run_and_log,infer,if [ "$$(UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run python -c "import sys, logging; logging.disable(logging.CRITICAL); from src.schemas import ExperimentConfig; config = ExperimentConfig.from_yaml(sys.argv[1]); sys.stdout.write(config.inference.backend)" $(CONFIG))" = "blackjax" ]; then $(UV) run python -m src.models.irt_blackjax --config $(CONFIG); else $(UV) run python -m src.models.irt_numpyro --config $(CONFIG); fi)

infer-blackjax:
	@$(call run_and_log,infer_blackjax,$(UV) run python -m src.models.irt_blackjax --config $(CONFIG))

diagnostics:
	@$(call run_and_log,diagnostics,$(UV) run python -m src.analysis.diagnostics --config $(CONFIG))

plots:
	@$(call run_and_log,plots,$(UV) run python -m src.analysis.plots --config $(CONFIG))

notebook:
	@$(call run_and_log,notebook,$(UV) run marimo edit notebooks/results.py)

full:
	@$(MAKE) items CONFIG=$(CONFIG)
	@$(MAKE) judge CONFIG=$(CONFIG)
	@$(MAKE) matrix CONFIG=$(CONFIG)
	@$(MAKE) validate CONFIG=$(CONFIG)
	@$(MAKE) infer CONFIG=$(CONFIG)
	@$(MAKE) diagnostics CONFIG=$(CONFIG)
	@$(MAKE) plots CONFIG=$(CONFIG)
