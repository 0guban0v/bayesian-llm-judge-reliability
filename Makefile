SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:

include mk/common.mk
include mk/quality.mk
include mk/pipeline.mk
include mk/profiling.mk
