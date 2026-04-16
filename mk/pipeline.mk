.PHONY: recommend-models verify-models setup-models items items-refresh judge matrix validate infer diagnostics plots report-exports notebook full

recommend-models:
	@$(call run_and_log,recommend_models,bash scripts/recommend_models.sh)

verify-models:
	@$(call run_and_log,verify_models,$(UV) run python scripts/verify_models.py --config $(CONFIG) $(MODELS))

setup-models:
	@$(call run_and_log,setup_models,$(UV) run python scripts/setup_models.py --config $(CONFIG))

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
	@$(call run_and_log,infer,$(UV) run python -m src.models.infer --config $(CONFIG))

diagnostics:
	@$(call run_and_log,diagnostics,$(UV) run python -m src.analysis.diagnostics --config $(CONFIG))

plots:
	@$(call run_and_log,plots,$(UV) run python -m src.analysis.plots --config $(CONFIG))

report-exports:
	@$(call run_and_log,report_exports,$(UV) run python -m src.analysis.report_exports --config $(CONFIG))

notebook:
	@$(UV) run marimo edit notebooks/results.py

full:
	@$(MAKE) items CONFIG=$(CONFIG)
	@$(MAKE) judge CONFIG=$(CONFIG)
	@$(MAKE) matrix CONFIG=$(CONFIG)
	@$(MAKE) validate CONFIG=$(CONFIG)
	@$(MAKE) infer CONFIG=$(CONFIG)
	@$(MAKE) diagnostics CONFIG=$(CONFIG)
	@$(MAKE) plots CONFIG=$(CONFIG)
	@$(MAKE) report-exports CONFIG=$(CONFIG)
