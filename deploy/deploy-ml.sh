#!/usr/bin/env bash
set -Eeuo pipefail

# Runs on the production host after building the image for DEPLOY_SHA.
: "${DEPLOY_ROOT:?Set DEPLOY_ROOT}"
: "${DEPLOY_SHA:?Set DEPLOY_SHA}"
: "${ML_SERVICE_TOKEN:?Set the ML_SERVICE_TOKEN GitHub Secret}"
[[ "$DEPLOY_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid commit SHA' >&2; exit 1; }
[[ "$ML_SERVICE_TOKEN" =~ ^[A-Za-z0-9_-]{32,256}$ ]] || { echo 'Use a 32-256 character URL-safe token' >&2; exit 1; }
[[ "$DEPLOY_ROOT" = /* && "$DEPLOY_ROOT" != / ]] || { echo 'Invalid deployment directory' >&2; exit 1; }
[[ "$(git rev-parse HEAD)" == "$DEPLOY_SHA" ]] || { echo 'Checked-out revision differs from CI revision' >&2; exit 1; }
command -v flock >/dev/null

umask 077
mkdir -p "$DEPLOY_ROOT/releases"
exec 9>"$DEPLOY_ROOT/.deploy.lock"
flock -w 300 9

docker network inspect backend >/dev/null
docker image inspect "hack-autumn-2026-ml:$DEPLOY_SHA" >/dev/null

# Each release keeps its own immutable Compose and environment snapshot.
release=$(mktemp -d "$DEPLOY_ROOT/releases/${DEPLOY_SHA}.XXXXXX")
previous=$(readlink -f "$DEPLOY_ROOT/current" 2>/dev/null || true)
cp compose.ml.production.yaml "$release/compose.ml.production.yaml"
cp inference_service/artifacts/manifest.json "$release/manifest.json"
printf 'ML_IMAGE_TAG=%s\nML_SERVICE_TOKEN=%s\n' "$DEPLOY_SHA" "$ML_SERVICE_TOKEN" > "$release/runtime.env"
chmod 600 "$release/runtime.env"
unset ML_SERVICE_TOKEN ML_IMAGE_TAG

dc() {
  local dir=$1
  shift
  env -u ML_SERVICE_TOKEN -u ML_IMAGE_TAG docker compose -p aeza-ml \
    --env-file "$dir/runtime.env" -f "$dir/compose.ml.production.yaml" "$@"
}

start_and_check() {
  local dir=$1 version digest
  version=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["modelVersion"])' "$dir/manifest.json") || return 1
  digest=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["modelSha256"])' "$dir/manifest.json") || return 1

  dc "$dir" up -d --no-build --wait --wait-timeout 120 || return 1
  dc "$dir" exec -T ml-service python - "$version" "$digest" <<'PY' || return 1
import json
import os
import sys
import urllib.request

request = urllib.request.Request(
    'http://127.0.0.1:8000/internal/v1/model',
    headers={'Authorization': 'Bearer ' + os.environ['ML_SERVICE_TOKEN']},
)
with urllib.request.urlopen(request, timeout=5) as response:
    info = json.load(response)
assert info['modelVersion'] == sys.argv[1], 'Wrong model version'
assert info['modelSha256'] == sys.argv[2], 'Wrong model hash'
assert set(info['supportedPaymentTerms']) == {'hour', 'day', 'month'}
print('ML ready:', info['modelVersion'])
PY
}

if start_and_check "$release" &&
   ln -sfn "$release" "$DEPLOY_ROOT/.current-next" &&
   mv -Tf "$DEPLOY_ROOT/.current-next" "$DEPLOY_ROOT/current"; then
  printf 'ML deployment succeeded: %s\n' "$DEPLOY_SHA"
else
  rm -f "$DEPLOY_ROOT/.current-next" || true
  echo 'ML deployment failed; attempting rollback' >&2
  if [[ -n "$previous" && -f "$previous/runtime.env" ]]; then
    if start_and_check "$previous"; then
      echo 'Previous ML release restored' >&2
    else
      echo 'Rollback failed; operator intervention required' >&2
    fi
  else
    dc "$release" rm -s -f ml-service || true
    echo 'First deployment failed; no previous release exists' >&2
  fi
  exit 1
fi
