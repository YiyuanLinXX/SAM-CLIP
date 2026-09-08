#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
target="${project_dir}/.env"
template="${project_dir}/.env.example"

if [[ -e "${target}" ]]; then
  echo ".env already exists; leaving it unchanged." >&2
  echo "Remove it first if you want to regenerate it." >&2
  exit 1
fi

sed \
  -e "s/^HOST_UID=.*/HOST_UID=$(id -u)/" \
  -e "s/^HOST_GID=.*/HOST_GID=$(id -g)/" \
  "${template}" > "${target}"

echo "Created ${target} with HOST_UID=$(id -u) and HOST_GID=$(id -g)."
echo "Edit the mounted data, weights, checkpoints, and outputs paths if needed."
