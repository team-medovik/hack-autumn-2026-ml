#!/usr/bin/env python3
"""Deploy only the generator; retain PostgreSQL and never disclose runtime secrets."""
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import tempfile

ROOT = Path('/home/kinoko/hack-autumn-2026/generator')
PROJECT = 'aeza-generator'
API = 'hack-autumn-2026-api'
PROFILES = {'stored', 'demo-stream-v1', 'demo-stream-v2'}


class DeploymentError(Exception):
    pass


def run(args, *, data=None, timeout=60):
    try:
        result = subprocess.run(args, input=data, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        raise DeploymentError('Command unavailable or timed out') from None
    if result.returncode:
        raise DeploymentError('Command failed: ' + args[0])
    return result.stdout


def inspect(name):
    return json.loads(run(['docker', 'inspect', name]))[0]


def env_of(container):
    return dict(item.split('=', 1) for item in container['Config'].get('Env', []) if '=' in item)


def private_write(path, value):
    with open(path, 'x', encoding='utf-8') as file:
        os.chmod(path, 0o600)
        file.write(value)


def dotenv(text):
    result = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            raise DeploymentError('Managed runtime.env contains an invalid line')
        key, value = line.split('=', 1)
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*', key) or '\n' in value or '\r' in value:
            raise DeploymentError('Managed runtime.env contains an invalid entry')
        result[key] = value
    return result


def render_dotenv(values):
    return ''.join(key + '=' + values[key] + '\n' for key in sorted(values))


PROBE = r'''
import json, sys, time, urllib.request
request = json.load(sys.stdin)
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
def status():
    req = urllib.request.Request('http://generator-api:8000/internal/v1/generator/status',
        headers={'Authorization': 'Bearer ' + request['token']})
    with opener.open(req, timeout=10) as response:
        assert response.status == 200
        body = json.load(response)
    assert body['syntheticOnly'] is True
    assert body['workerState'] in ('starting', 'running', 'finished')
    return body
try:
    before = status()
    if request.get('baselineOnly'):
        print(json.dumps(before['generator']))
        sys.exit(0)
    baseline = request.get('baseline')
    if baseline:
        current = before['generator']
        assert current['runId'] == baseline['runId']
        assert current['revision'] >= baseline['revision'] and current['eventCount'] >= baseline['eventCount']
        assert current['virtualTime'] >= baseline['virtualTime']
    minimum_events = request.get('minimumEvents', 0)
    require_event_increase = request.get('requireEventIncrease', False)
    deadline = time.monotonic() + 45
    while True:
        time.sleep(2)
        after = status()
        a, b = before['generator'], after['generator']
        assert a['runId'] == b['runId']
        assert b['revision'] >= a['revision'] and b['eventCount'] >= a['eventCount']
        assert b['virtualTime'] >= a['virtualTime']
        event_increased = not require_event_increase or not baseline or b['eventCount'] > baseline['eventCount']
        if b['eventCount'] >= minimum_events and event_increased and (after['workerState'] == 'finished' or
                (after['workerState'] == 'running' and b['virtualTime'] > a['virtualTime'])):
            print(json.dumps({'statusHttp': 200, 'runStable': True, 'progressVerified': True}))
            break
        assert time.monotonic() < deadline
except Exception:
    print('Generator status/progress verification failed', file=sys.stderr)
    sys.exit(1)
'''

def main():
    os.umask(0o077)
    sha = os.environ.get('GITHUB_SHA', '')
    profile = os.environ.get('GENERATOR_DEPLOY_PROFILE', 'stored')
    if profile not in PROFILES:
        raise DeploymentError('Unknown generator deployment profile')
    if not re.fullmatch('[0-9a-f]{40}', sha) or run(['git', 'rev-parse', 'HEAD']).strip() != sha:
        raise DeploymentError('Checked-out revision must match GITHUB_SHA')
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / '.deploy.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        api = inspect(API)
        if api['State']['Status'] != 'running' or 'backend' not in api['NetworkSettings']['Networks']:
            raise DeploymentError('Running API must be connected to backend')
        token = env_of(api).get('APP_GENERATOR_SERVICE_TOKEN', '')
        if not re.fullmatch(r'[A-Za-z0-9_./+=-]{16,512}', token):
            raise DeploymentError('API service token is missing or has unsupported encoding')
        run(['docker', 'network', 'inspect', 'backend'])
        # Never silently introduce a second generator project beside an existing run.
        existing = []
        existing_by_service = {}
        for service in ('generator-api', 'generator-worker', 'generator-db'):
            ids = run(['docker', 'ps', '-aq', '--filter', 'label=com.docker.compose.service=' + service]).split()
            existing.extend(ids)
            for container_id in ids:
                labels = inspect(container_id)['Config'].get('Labels') or {}
                if labels.get('com.docker.compose.project') != PROJECT:
                    raise DeploymentError('Existing generator uses another Compose project; preserve it explicitly')
                if inspect(container_id)['State']['Status'] == 'running':
                    existing_by_service[service] = inspect(container_id)
        runtime = ROOT / 'runtime.env'
        if not runtime.exists():
            volumes = run(['docker', 'volume', 'ls', '--format', '{{.Name}}']).split()
            if existing or any('generator-postgres' in name or name.startswith(PROJECT + '_') for name in volumes):
                raise DeploymentError('Existing generator state has no managed runtime.env; refusing to replace credentials')
            private_write(runtime, 'GENERATOR_POSTGRES_PASSWORD=' + secrets.token_urlsafe(32) + '\n'
                'GENERATOR_RUN_KEY=default\nGENERATOR_TIME_SCALE=60\nGENERATOR_TICK_SECONDS=1\n'
                'GENERATOR_MAX_STEP_SECONDS=3600\nGENERATOR_START=2026-01-01T00:00:00Z\n'
                'GENERATOR_DAYS=365\nGENERATOR_ACCOUNTS=50\nGENERATOR_SEED=20261024\n'
                'GENERATOR_PAYMENT_TERMS=month\n')
        if runtime.is_symlink() or runtime.stat().st_mode & 0o077:
            raise DeploymentError('runtime.env must be a private regular file (mode 0600)')
        target_environment = dotenv(runtime.read_text())
        target_run_key = target_environment.get('GENERATOR_RUN_KEY', 'default')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', target_run_key):
            raise DeploymentError('Target generator run key is invalid')
        image = 'hack-autumn-2026-generator:' + sha
        print('Building generator image for revision ' + sha, flush=True)
        run(['docker', 'build', '-f', 'generator_service/Dockerfile', '-t', image, '.'], timeout=900)
        releases = ROOT / 'releases'
        releases.mkdir(exist_ok=True)
        release = Path(tempfile.mkdtemp(prefix=sha + '.', dir=releases))
        shutil.copyfile('compose.generator.production.yaml', release / 'compose.yaml')
        target_environment.update({'GENERATOR_IMAGE_TAG': sha,
            'GENERATOR_SERVICE_TOKEN': token, 'LIVE_STACK_NETWORK': 'backend'})
        private_write(release / 'runtime.env', render_dotenv(target_environment))
        compose = ['docker', 'compose', '-p', PROJECT, '--env-file', str(release / 'runtime.env'),
                   '-f', str(release / 'compose.yaml')]
        config = json.loads(run(compose + ['config', '--format', 'json']))
        services = config['services']
        if set(services) != {'generator-api', 'generator-worker', 'generator-db'}:
            raise DeploymentError('Unexpected services in generator Compose')
        if any(service.get('ports') for service in services.values()):
            raise DeploymentError('Generator deployment must not publish host ports')
        if services['generator-api']['environment']['GENERATOR_SERVICE_TOKEN'] != token:
            raise DeploymentError('Resolved service token differs from running API')
        probe = ['docker', 'run', '--rm', '-i', '--network', 'container:' + API,
                 '--entrypoint', 'python', image, '-B', '-c', PROBE]
        baseline = None
        if existing:
            baseline = json.loads(run(probe, data=json.dumps({'token': token, 'baselineOnly': True}), timeout=30))
        stopped = []
        try:
            if profile == 'demo-stream-v1':
                if not all(service in existing_by_service for service in ('generator-api', 'generator-worker', 'generator-db')):
                    raise DeploymentError('Demo schedule migration requires the complete running generator stack')
                stopped = [existing_by_service[name]['Id'] for name in ('generator-worker', 'generator-api')]
                run(['docker', 'stop'] + stopped, timeout=45)
                migration = compose + ['run', '--rm', '--no-deps', 'generator-worker',
                    'python', '-B', '-m', 'generator_service.repace', '--apply', '--marker', profile,
                    '--spread-days', '60', '--warmup-seconds', '1']
                migration_result = json.loads(run(migration, timeout=90))
                if migration_result.get('status') not in ('applied', 'already_applied'):
                    raise DeploymentError('Unexpected demo schedule migration result')
            print('Starting generator profile ' + profile + ' with PostgreSQL state retained', flush=True)
            run(compose + ['up', '-d', '--no-build', '--wait', '--wait-timeout', '180'], timeout=210)
            if profile == 'demo-stream-v2':
                enable = compose + ['exec', '-T', 'generator-worker', 'python', '-B', '-m',
                    'generator_service.stream', '--apply', '--marker', 'demo-stream-72h-v2',
                    '--min-real-seconds', '10', '--max-real-seconds', '30',
                    '--services-per-client', '6', '--minimum-real-hours', '72']
                enabled = json.loads(run(enable, timeout=90))
                if enabled.get('status') not in ('enabled', 'already_enabled'):
                    raise DeploymentError('Unexpected rolling demo stream result')
            # A temporary probe shares the actual API network namespace, not the generator's.
            output = run(probe, data=json.dumps({'token': token, 'baseline': baseline,
                'minimumEvents': 3 if profile != 'stored' else 0,
                'requireEventIncrease': profile == 'demo-stream-v2'}), timeout=80)
            result = json.loads(output)
            if result != {'statusHttp': 200, 'runStable': True, 'progressVerified': True}:
                raise DeploymentError('Unexpected probe result')
            for container_id in run(compose + ['ps', '-q']).split():
                if any(inspect(container_id)['NetworkSettings'].get('Ports', {}).values()):
                    raise DeploymentError('Generator container unexpectedly publishes a host port')
            next_link = ROOT / '.current-next'
            next_link.unlink(missing_ok=True)
            next_link.symlink_to(release)
            next_link.replace(ROOT / 'current')
        except Exception:
            if stopped:
                try:
                    subprocess.run(['docker', 'start'] + stopped, capture_output=True, text=True, timeout=45)
                except Exception:
                    pass
            raise
        print('Generator deployment verified: HTTP 200 from API network, progress stable, no host ports', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Docker errors, Python tracebacks and environment values may contain credentials.
        print(str(error) if isinstance(error, DeploymentError) else 'Generator deployment failed; runtime data retained', flush=True)
        raise SystemExit(1)
