.PHONY: sync lint format dead-code quality test pre-commit-install pre-commit-run

sync:
	@$(UV) sync

lint:
	@$(UV) run ruff check src tests notebooks scripts --fix

format:
	@$(UV) run ruff format src tests notebooks scripts

dead-code:
	@$(UV) run vulture src tests tests/vulture_whitelist.py

quality:
	@$(MAKE) lint
	@$(MAKE) format
	@$(MAKE) dead-code

test:
	@UV_CACHE_DIR=$(UV_CACHE_DIR) \
		MPLCONFIGDIR=$(MPLCONFIGDIR) \
		$(WITH_CACHE_ENV) $(UV) run python -m unittest discover -s tests

pre-commit-install:
	@$(UV) run pre-commit install --hook-type pre-commit --hook-type pre-push

pre-commit-run:
	@UV_CACHE_DIR=$(UV_CACHE_DIR) \
		MPLCONFIGDIR=$(MPLCONFIGDIR) \
		$(WITH_CACHE_ENV) $(UV) run pre-commit run --all-files
