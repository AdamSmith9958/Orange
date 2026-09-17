#!/usr/bin/env bash
# Point the local test environment at a pack under data/ (writes LOCAL_ENV_PACK_PATH/SHA256 into .env).
#
#   ./set_pack.sh                 # uses PACK below
#   ./set_pack.sh data/race1      # a pack directory (holds env-pack.tar.gz) or the .tar.gz itself
set -euo pipefail

PACK="data/local-test"

cd "$(dirname "$0")"
PACK="${1:-$PACK}"
[[ -d "$PACK" ]] && PACK="$PACK/env-pack.tar.gz"
if [[ ! -f "$PACK" ]]; then
  echo "pack not found: $PACK" >&2
  exit 1
fi

path="/workspace/${PACK#./}"
sha="$(sha256sum "$PACK" | cut -d' ' -f1)"

touch .env
sed -i '/^LOCAL_ENV_PACK_PATH=/d; /^LOCAL_ENV_PACK_SHA256=/d' .env
printf 'LOCAL_ENV_PACK_PATH=%s\nLOCAL_ENV_PACK_SHA256=%s\n' "$path" "$sha" >> .env

echo "LOCAL_ENV_PACK_PATH=$path"
echo "LOCAL_ENV_PACK_SHA256=$sha"
