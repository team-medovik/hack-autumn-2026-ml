# Что настроить на сервере для ML

Эта инструкция относится к одному Linux-серверу с Kotlin API, Docker Engine и
Nginx. ML-деплой из `.github/workflows/deploy-ml.yml` выполняется после успешного
`ML basic checks` для `main`. До попадания workflow в `main` автоматического
деплоя нет. Генератор живёт отдельно и этим workflow не обновляется.

## Адреса и маршруты

| Кто обращается | Адрес | Назначение |
|---|---|---|
| Kotlin API в Docker-сети `backend` | `http://ml-service:8000` | Внутренний ML API |
| Администратор на самом сервере | `http://127.0.0.1:8012/health/ready` | Проверка готовности модели без токена |
| Kotlin API в Docker-сети `backend` при запущенном генераторе | `http://generator-api:8000` | Внутренний API генератора |
| Администратор на самом сервере, если запущен генератор | `http://127.0.0.1:8011/health/ready` | Проверка генератора без токена |
| Nginx на том же сервере | `http://127.0.0.1:33000` | Публичные маршруты Kotlin `/api/v1/*` |
| Nginx | `/home/kinoko/hack-autumn-2026/front` | Статические файлы фронта |

У ML есть `GET /health/live`, `GET /health/ready`, защищённый Bearer-токеном
`GET /internal/v1/model` и защищённый `POST /internal/v1/renewal-risk/predict`.
У генератора есть свои `/health/*` и защищённые
`/internal/v1/generator/status`, `/events`, `/snapshot`. Эти внутренние маршруты
не нужны браузеру и не требуют отдельного публичного Nginx location.

Указанный в задаче `compose.ml.yaml` создаёт отдельную сеть `ml-internal` и
предназначен для локального запуска. В production используйте
`compose.ml.production.yaml`: он подключает ML к существующей внешней сети
`backend`. Не запускайте оба Compose одновременно: они претендуют на один
хостовый порт `8012`.

## Один раз подготовить сервер

Установите Docker Engine, Compose v2 с поддержкой `up --wait`, Git, Bash,
Python 3, GNU coreutils (`readlink -f`, `mv -T`) и `flock`. Пользователь runner
должен иметь доступ к Docker daemon. На сервере нужны доступ к GitHub,
Docker Hub и PyPI для checkout и сборки образа, а также место минимум для двух
ML-образов и релизов.

```bash
docker version
docker compose version
docker compose up --help | grep -- --wait
python3 --version
command -v flock
docker network inspect backend
```

Если `backend` ещё нет именно на сервере API, создайте её один раз:

```bash
docker network create backend
```

Зарегистрируйте Linux self-hosted runner **в ML-репозитории** через
`Settings → Actions → Runners → New self-hosted runner`, добавьте метку `deploy`
и установите runner как службу. Job требует метки
`[self-hosted, linux, deploy]`. Runner API или фронта сам по себе ML-репозиторию
недоступен. В примере ниже сервис runner работает от пользователя `kinoko`;
если он другой, подставьте его имя. Каталог API этот workflow не очищает.

```bash
DEPLOY_USER=kinoko
sudo install -d -m 700 -o "$DEPLOY_USER" -g "$(id -gn "$DEPLOY_USER")" \
  /home/kinoko/hack-autumn-2026/ml \
  /home/kinoko/hack-autumn-2026/ml/releases
docker network inspect backend >/dev/null
```

Runner не должен обслуживать недоверенные PR: PR-проверки идут на
`ubuntu-latest`, а production job запускается только после успешного CI для
`main`. Не удаляйте старые ML-образы глобальным `docker system prune -a`:
они нужны для отката.

## Секреты и параметры

В **ML-репозитории** нужен ровно один Actions repository secret:

| Имя | Значение |
|---|---|
| `ML_SERVICE_TOKEN` | Случайная строка из 32–256 символов `A–Z`, `a–z`, `0–9`, `_`, `-`. Например, 64 hex-символа. |

Создать: `Settings → Secrets and variables → Actions → New repository secret`.
Это межсервисный Bearer-токен, не JWT пользователя. Workflow сам создаёт
закрытый `runtime.env` внутри каталога релиза. В ML-репозитории **не нужны**
`ENV_PRODUCTION`, MinIO credentials, SSH key или публичный URL модели.

Остальные настройки хранятся в своих местах:

| Где | Что положить |
|---|---|
| API-репозиторий, secret `ENV_PRODUCTION` | Существующие параметры API плюс `APP_ML_*`; при включении генератора — `APP_GENERATOR_*` |
| Front-репозиторий, secret `ENV_PRODUCTION` | Публичный `VITE_API_URL` и остальные параметры сборки фронта |
| Сервер, `generator/generator.env` | Пароль PostgreSQL, токен генератора и параметры запуска; только если нужен генератор |

Значение `ML_SERVICE_TOKEN` нужно сохранить в менеджере секретов: его точная
копия понадобится в `APP_ML_SERVICE_TOKEN` у Kotlin API. Не вставляйте токен в
Git, PR, Nginx-конфиг, командную строку `curl` или логи. При смене токена
обновите обе стороны согласованно.

API-репозиторий уже собирает `.env` из своего Actions secret `ENV_PRODUCTION`.
После выпуска [API PR #42 — интеграции генератора и ML](https://github.com/team-medovik/hack-autumn-2026-api/pull/42)
дополните **существующее** значение этого секрета; не заменяйте файл целиком и
не правьте только `.env` на сервере: следующий деплой API его перезапишет.

```dotenv
APP_ML_ENABLED=true
APP_ML_BASE_URL=http://ml-service:8000
APP_ML_SERVICE_TOKEN=<то же значение, что ML_SERVICE_TOKEN>
APP_ML_MODEL_VERSION=event-driven-all-terms-v1-234422ba1357
APP_ML_REQUEST_TIMEOUT=2s
APP_ML_CACHE_TTL=30s
APP_ML_MAX_HISTORY_EVENTS=100000
```

`APP_ML_MODEL_VERSION` должен совпадать с `modelVersion` в
`inference_service/artifacts/manifest.json`. ML по умолчанию отключён в API;
пока PR интеграции не выпущен в `main`, настройка env сама по себе не добавит
вызовы модели. Оба контейнера должны находиться в одной сети `backend`.

## Nginx

Для обычной схемы внешний HTTPS-адрес обслуживает фронт и Kotlin API. Файл
фронта лежит в `/home/kinoko/hack-autumn-2026/front`; дайте пользователю Nginx
право читать файлы и проходить родительские каталоги. Внутри существующего
TLS `server` блока можно использовать:

```nginx
root /home/kinoko/hack-autumn-2026/front;
index index.html;

location /api/ {
    proxy_pass http://127.0.0.1:33000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

location = /api/v1/events {
    proxy_pass http://127.0.0.1:33000;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;
    proxy_read_timeout 1h;
}

location / {
    try_files $uri $uri/ /index.html;
}
```

У `proxy_pass` намеренно нет завершающего `/`: так Nginx сохраняет путь
`/api/v1/...`. TLS-сертификат, `server_name` и редирект HTTP → HTTPS настройте
в существующем серверном конфиге. После изменения выполните:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

Публичный пример: `https://<ваш-домен>/api/v1/...`.

Порт API `33000` и management-порт `33001` в текущем API Compose опубликованы
на всех интерфейсах. Если Nginx на том же хосте, ограничьте их firewall или
измените привязку в **API-репозитории** на `127.0.0.1:33000:33000` и
`127.0.0.1:33001:33001` с отдельным релизом API. Ручная правка Compose на
сервере будет затёрта его workflow. ML `8012` и генератор `8011` уже
привязаны к loopback. С другого хоста эти адреса недоступны; для отдельного
Nginx/API-хоста потребуется защищённая сетевая связность и другая адресация.

Фронтенд собирается со своим `ENV_PRODUCTION` и `VITE_API_URL`. Указывайте в
нём публичный HTTPS-адрес API, согласованный с путями запросов фронта, а не
`ml-service:8000` и не серверный loopback-адрес.

## Если нужен live-генератор

Он не входит в ML workflow. Для него нужен стабильный checkout вне каталогов,
которые очищает API-деплой, отдельная PostgreSQL-БД и постоянный volume:

```bash
mkdir -p /home/kinoko/hack-autumn-2026/generator
git clone https://github.com/team-medovik/hack-autumn-2026-ml.git \
  /home/kinoko/hack-autumn-2026/generator/source
git -C /home/kinoko/hack-autumn-2026/generator/source switch main
```

В `/home/kinoko/hack-autumn-2026/generator/generator.env` с правами `600`
задайте `GENERATOR_POSTGRES_PASSWORD`, `GENERATOR_SERVICE_TOKEN`, неизменный
`GENERATOR_RUN_KEY` и нужные параметры симуляции. Не используйте дефолтный
пароль `generator-local`. Секрет генератора и пароль БД должны быть разными.
Для встроенной БД не задавайте `GENERATOR_DATABASE_URL`: Compose составит
его сам. Существующий запуск не пересоздаётся при изменении seed или времени.

```dotenv
GENERATOR_POSTGRES_PASSWORD=<отдельный случайный пароль>
GENERATOR_SERVICE_TOKEN=<отдельный случайный токен>
GENERATOR_RUN_KEY=demo-v1
GENERATOR_START=2026-01-01T00:00:00Z
GENERATOR_DAYS=365
GENERATOR_ACCOUNTS=50
GENERATOR_SEED=20261024
GENERATOR_TIME_SCALE=60
GENERATOR_TICK_SECONDS=1
GENERATOR_MAX_STEP_SECONDS=3600
```

Замените оба значения в угловых скобках до запуска. Для пароля и токена удобно
использовать разные строки из 64 hex-символов, чтобы DSN не потребовал
URL-кодирования. Для приватного репозитория сначала настройте read-only доступ
Git на сервере; не добавляйте токен в URL `git clone`.

Для подключения только `generator-api` к `backend` положите рядом с env-файлом
override `compose.generator.production.yaml`:

```yaml
services:
  generator-db:
    environment:
      POSTGRES_PASSWORD: ${GENERATOR_POSTGRES_PASSWORD:?Set a real database password}
  generator-api:
    networks: [generator-internal, backend]
networks:
  backend:
    external: true
    name: backend
```

При запуске всегда используйте один project name `aeza-generator` и оба
Compose-файла: иначе Docker создаст новый volume и старое состояние окажется
не видно. Не выполняйте `down -v`.

```bash
chmod 600 /home/kinoko/hack-autumn-2026/generator/generator.env
docker compose -p aeza-generator \
  --env-file /home/kinoko/hack-autumn-2026/generator/generator.env \
  -f /home/kinoko/hack-autumn-2026/generator/source/compose.generator.yaml \
  -f /home/kinoko/hack-autumn-2026/generator/compose.generator.production.yaml \
  up -d --build --wait --wait-timeout 180
curl --fail http://127.0.0.1:8011/health/ready
```

В `ENV_PRODUCTION` API после выпуска того же PR #42, если включаете генератор, нужны
`APP_GENERATOR_BASE_URL=http://generator-api:8000`,
`APP_GENERATOR_SERVICE_TOKEN` с тем же значением, что серверный
`GENERATOR_SERVICE_TOKEN`, и включённые флаги `APP_GENERATOR_ENABLED` /
`APP_GENERATOR_USE_FOR_DASHBOARD`. Пароль БД генератора в API не нужен.

## Проверка после деплоя

```bash
curl --fail http://127.0.0.1:8012/health/ready
ML_RELEASE=$(readlink -f /home/kinoko/hack-autumn-2026/ml/current)
docker compose -p aeza-ml --env-file "$ML_RELEASE/runtime.env" \
  -f "$ML_RELEASE/compose.ml.production.yaml" ps
docker network inspect backend --format '{{range .Containers}}{{println .Name}}{{end}}'
```

В сети должны быть Kotlin API и `ml-service`; при включённом генераторе ещё
`generator-api`. `curl` к `/health/ready` должен вернуть HTTP 200 и
`{"status":"ready"}`. В GitHub Actions деплой должен закончиться успешной
проверкой версии/SHA. Если job висит в очереди, проверьте runner и метки;
если модель не готова, смотрите логи `aeza-ml` и manifest. Предыдущий релиз
остаётся в `/home/kinoko/hack-autumn-2026/ml/releases` для отката. При самом
первом неуспешном деплое предыдущего релиза ещё нет.
