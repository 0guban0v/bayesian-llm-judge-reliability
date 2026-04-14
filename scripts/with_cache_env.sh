#!/usr/bin/env bash

set -euo pipefail

cache_root="${UV_CACHE_DIR:-.uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$cache_root/matplotlib}"
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:-compiledir=$cache_root/pytensor}"

mkdir -p "$MPLCONFIGDIR" "$cache_root/pytensor"

exec "$@"
