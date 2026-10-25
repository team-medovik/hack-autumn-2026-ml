"""Validate a Compose JSON stream without exposing secrets or sourcing dotenv."""
import json
import re
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    config = json.load(sys.stdin)
    kind = sys.argv[1]
    services = config['services']
    token = services['ml-service' if kind == 'ml' else 'generator-api']['environment'][
        'ML_SERVICE_TOKEN' if kind == 'ml' else 'GENERATOR_SERVICE_TOKEN']
    require(isinstance(token, str) and len(token) >= 16 and '\r' not in token and '\n' not in token,
            'Service token must contain at least 16 characters without line breaks')
    if kind == 'generator':
        password = services['generator-db']['environment']['POSTGRES_PASSWORD']
        require(isinstance(password, str) and re.fullmatch(r'[A-Za-z0-9._~-]+', password)
                and password != 'generator-local',
                'Set a non-default URI-safe PostgreSQL password')
        api = services['generator-api']['environment']
        worker = services['generator-worker']['environment']
        require(api['GENERATOR_DATABASE_URL'] == worker['GENERATOR_DATABASE_URL'] and
                api['GENERATOR_RUN_KEY'] == worker['GENERATOR_RUN_KEY'],
                'Generator API and worker must share database and run key')
    print(kind + ' configuration valid')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError):
        # Never interpolate parsed data: this process receives resolved secrets.
        print('Invalid live stack configuration; check required secrets and settings', file=sys.stderr)
        sys.exit(1)
