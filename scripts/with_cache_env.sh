#!/usr/bin/env bash

set -euo pipefail

cache_root="${UV_CACHE_DIR:-.uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$cache_root/matplotlib}"

mkdir -p "$MPLCONFIGDIR" "$cache_root/pytensor"

existing_pytensor_flags="${PYTENSOR_FLAGS:-}"
if [[ "$existing_pytensor_flags" != *"compiledir="* ]]; then
  if [[ -n "$existing_pytensor_flags" ]]; then
    export PYTENSOR_FLAGS="$existing_pytensor_flags,compiledir=$cache_root/pytensor"
  else
    export PYTENSOR_FLAGS="compiledir=$cache_root/pytensor"
  fi
fi

exec "$@"
