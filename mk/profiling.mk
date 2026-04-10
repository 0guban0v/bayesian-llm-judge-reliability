.PHONY: profile-judge profile-matrix profile-infer profile-plots profile-full

profile-judge:
	@mkdir -p $(PROFILE_DIR)
	@profile_path="$(PROFILE_DIR)/judge_$$(date +'%Y%m%d_%H%M%S').speedscope.json"
	@echo "writing $$profile_path"
	@$(UV) run python scripts/profile_command.py --target judge --profile-path "$$profile_path" --config $(CONFIG) -- $(PY_SPY) record --format speedscope -o "$$profile_path" -- $(UV) run python -m src.judges.runner --config $(CONFIG) $(if $(JUDGE),--judge $(JUDGE),) $(if $(LIMIT),--limit $(LIMIT),)

profile-matrix:
	@mkdir -p $(PROFILE_DIR)
	@profile_path="$(PROFILE_DIR)/matrix_$$(date +'%Y%m%d_%H%M%S').speedscope.json"
	@echo "writing $$profile_path"
	@$(UV) run python scripts/profile_command.py --target matrix --profile-path "$$profile_path" --config $(CONFIG) -- $(PY_SPY) record --format speedscope -o "$$profile_path" -- $(UV) run python -m src.data.loader --config $(CONFIG) --rebuild-matrix

profile-infer:
	@mkdir -p $(PROFILE_DIR)
	@profile_path="$(PROFILE_DIR)/infer_$$(date +'%Y%m%d_%H%M%S').speedscope.json"
	@echo "writing $$profile_path"
	@if [ "$$(UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run python -c "import sys, logging; logging.disable(logging.CRITICAL); from src.schemas import ExperimentConfig; config = ExperimentConfig.from_yaml(sys.argv[1]); sys.stdout.write(config.inference.backend)" $(CONFIG))" = "blackjax" ]; then \
		$(UV) run python scripts/profile_command.py --target infer --profile-path "$$profile_path" --config $(CONFIG) -- $(PY_SPY) record --format speedscope -o "$$profile_path" -- $(UV) run python -m src.models.irt_blackjax --config $(CONFIG); \
	else \
		$(UV) run python scripts/profile_command.py --target infer --profile-path "$$profile_path" --config $(CONFIG) -- $(PY_SPY) record --format speedscope -o "$$profile_path" -- $(UV) run python -m src.models.irt_numpyro --config $(CONFIG); \
	fi

profile-plots:
	@mkdir -p $(PROFILE_DIR)
	@profile_path="$(PROFILE_DIR)/plots_$$(date +'%Y%m%d_%H%M%S').speedscope.json"
	@echo "writing $$profile_path"
	@$(UV) run python scripts/profile_command.py --target plots --profile-path "$$profile_path" --config $(CONFIG) -- $(PY_SPY) record --format speedscope -o "$$profile_path" -- $(UV) run python -m src.analysis.plots --config $(CONFIG)

profile-full:
	@echo "writing profiles/metrics/full_<timestamp>.json"
	@$(UV) run python scripts/profile_command.py --target full --config $(CONFIG) -- $(MAKE) full CONFIG=$(CONFIG)
