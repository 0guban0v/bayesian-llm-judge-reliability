UV ?= uv
PY_SPY ?= $(UV) run py-spy
CONFIG ?= configs/experiment.yaml
JUDGE ?=
LIMIT ?=
LOG_DIR ?= logs
UV_CACHE_DIR ?= .uv-cache
PROFILE_DIR ?= profiles

export UV_PROJECT_ENVIRONMENT := .venv
export UV_CACHE_DIR

define log_path
$(LOG_DIR)/$(1)_$(shell date +'%Y%m%d_%H%M%S').log
endef

define run_and_log
	log_path="$(call log_path,$(1))"; \
	mkdir -p $(LOG_DIR); \
	if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
	echo "logging to $$log_path"; \
	$(2) 2>&1 | tee "$$log_path"
endef
