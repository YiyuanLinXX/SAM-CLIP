#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${project_dir}"

if [[ ! -f .env ]]; then
  echo "Note: .env not found; using paths from .env.example defaults." >&2
fi

exec docker compose run --rm sam-clip "$@"
