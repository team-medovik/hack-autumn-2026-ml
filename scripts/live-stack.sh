#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo 'Usage: scripts/live-stack.sh <validate|up|status|check|logs|stop> /path/to/.env.live'
  echo 'Defaults: projects aeza-generator / aeza-ml; pre-existing Docker network backend.'
  echo 'Isolation overrides (shell env): LIVE_STACK_NAMESPACE, LIVE_STACK_NETWORK.'
  echo 'Existing projects (shell env): LIVE_STACK_GENERATOR_PROJECT, LIVE_STACK_ML_PROJECT.'
}
if [[ $# != 2 ]]; then usage >&2; exit 2; fi
action=$1
case "$action" in validate|up|status|check|logs|stop) ;; *) usage >&2; exit 2 ;; esac
env_file=$2
[[ -f "$env_file" ]] || { echo 'Environment file does not exist' >&2; exit 2; }
env_file=$(cd "$(dirname "$env_file")" && printf '%s/%s' "$PWD" "$(basename "$env_file")")
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
namespace=${LIVE_STACK_NAMESPACE:-aeza}
[[ "$namespace" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || { echo 'Invalid LIVE_STACK_NAMESPACE' >&2; exit 2; }
export LIVE_STACK_NETWORK=${LIVE_STACK_NETWORK:-backend}
[[ "$LIVE_STACK_NETWORK" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid LIVE_STACK_NETWORK' >&2; exit 2; }
generator_project=${LIVE_STACK_GENERATOR_PROJECT:-$namespace-generator}
ml_project=${LIVE_STACK_ML_PROJECT:-$namespace-ml}
for project in "$generator_project" "$ml_project"; do
  [[ "$project" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || { echo 'Invalid Compose project override' >&2; exit 2; }
done
[[ "$generator_project" != "$ml_project" ]] || { echo 'Generator and ML need distinct Compose projects' >&2; exit 2; }
gen=(docker compose -p "$generator_project" --env-file "$env_file" -f "$root/compose.generator.production.yaml")
ml=(docker compose -p "$ml_project" --env-file "$env_file" -f "$root/compose.ml.production.yaml")

"${gen[@]}" config --format json | python3 "$root/scripts/check-live-config.py" generator
"${ml[@]}" config --format json | python3 "$root/scripts/check-live-config.py" ml
[[ "$action" != validate ]] || exit 0
case "$action" in up|check) docker network inspect "$LIVE_STACK_NETWORK" >/dev/null ;; esac
printf 'Projects: %s / %s; network: %s\n' "$generator_project" "$ml_project" "$LIVE_STACK_NETWORK"

check() {
  local version digest tick_seconds
  version=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["modelVersion"])' "$root/inference_service/artifacts/manifest.json")
  digest=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["modelSha256"])' "$root/inference_service/artifacts/manifest.json")
  "${ml[@]}" exec -T ml-service python - "$version" "$digest" <<'PY'
import json, os, sys, urllib.request
base = 'http://127.0.0.1:8000'
urllib.request.urlopen(base + '/health/ready', timeout=5).close()
request = urllib.request.Request(base + '/internal/v1/model', headers={
    'Authorization': 'Bearer ' + os.environ['ML_SERVICE_TOKEN']})
with urllib.request.urlopen(request, timeout=5) as response:
    info = json.load(response)
assert info['modelVersion'] == sys.argv[1], 'Model version differs from this checkout'
assert info['modelSha256'] == sys.argv[2], 'Model hash differs from this checkout'
print('ML ready:', info['modelVersion'], info['supportedPaymentTerms'])
PY
  # Read the running worker's interval, not a possibly edited dotenv file.
  tick_seconds=$("${gen[@]}" exec -T generator-worker python -c 'import os; print(os.environ.get("GENERATOR_TICK_SECONDS", "1"))')
  "${gen[@]}" exec -T generator-api python - "$tick_seconds" <<'PY'
import json, math, os, sys, time, urllib.request
tick_seconds = float(sys.argv[1])
assert math.isfinite(tick_seconds) and tick_seconds > 0, 'Invalid running worker interval'
base = 'http://127.0.0.1:8000'
urllib.request.urlopen(base + '/health/ready', timeout=5).close()
urllib.request.urlopen('http://ml-service:8000/health/ready', timeout=5).close()
def status():
    request = urllib.request.Request(base + '/internal/v1/generator/status', headers={
        'Authorization': 'Bearer ' + os.environ['GENERATOR_SERVICE_TOKEN']})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)
a = status()
deadline = time.monotonic() + max(10, tick_seconds * 3 + 5)
previous = a
while True:
    remaining = deadline - time.monotonic()
    assert remaining > 0, 'Virtual time did not advance within the worker progress window'
    time.sleep(min(1, remaining))
    b = status()
    assert a['generator']['runId'] == b['generator']['runId'], 'Generator run changed'
    assert b['workerState'] in ('starting', 'running'), 'Worker is not running (check logs or finished horizon)'
    assert b['generator']['virtualTime'] >= previous['generator']['virtualTime'], 'Virtual time moved backwards'
    assert b['generator']['eventCount'] >= previous['generator']['eventCount'], 'Event count moved backwards'
    if b['workerState'] == 'running' and b['generator']['virtualTime'] > a['generator']['virtualTime']:
        break
    previous = b
print('Generator live:', b['generator']['runId'], 'revision', b['generator']['revision'],
      'events', b['generator']['eventCount'], 'virtualTime', b['generator']['virtualTime'])
print('Shared network: generator-api reaches ml-service')
PY
}

case "$action" in
  up)
    # Builds complete before either service is recreated. No down/reset/volume removal.
    "${gen[@]}" build
    "${ml[@]}" build
    "${gen[@]}" up -d --no-build --wait --wait-timeout 180
    "${ml[@]}" up -d --no-build --wait --wait-timeout 120
    check
    ;;
  check) check ;;
  status) "${gen[@]}" ps; "${ml[@]}" ps ;;
  logs) "${gen[@]}" logs --tail 100; "${ml[@]}" logs --tail 100 ;;
  stop) "${ml[@]}" stop; "${gen[@]}" stop ;;
esac
