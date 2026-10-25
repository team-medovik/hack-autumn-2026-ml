# Постоянный генератор и HTTP-журнал

Генератор и ML независимы. Здесь нет загрузки CatBoost, обучения, расчёта риска
или вызовов модели. `generator-worker` выполняет биллинг и действия клиентов;
`generator-api` только читает результаты. Можно запускать несколько HTTP-процессов,
но на один checkpoint допускается ровно один worker. Чтение не запускает генерацию.

## Запуск

Python 3.13 и локальная POSIX-файловая система (Linux/macOS):

```sh
python3.13 -m venv .venv-generator
.venv-generator/bin/pip install -r generator_service/requirements.txt
export GENERATOR_CHECKPOINT="$PWD/runtime/live.json"
export GENERATOR_SERVICE_TOKEN="$(python3.13 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export GENERATOR_TIME_SCALE=3600

# Процесс 1: создаёт checkpoint только при его отсутствии, иначе продолжает.
.venv-generator/bin/python -m generator_service.worker

# Процесс 2: в другом терминале с теми же переменными окружения.
.venv-generator/bin/python -m uvicorn generator_service.api:create_app \
  --factory --host 127.0.0.1 --port 8011
```

Или `docker compose -f compose.generator.yaml up -d --build`, предварительно задав
`GENERATOR_SERVICE_TOKEN`. Named volume хранит состояние между рестартами контейнеров;
API монтирует его только для чтения. `restart: unless-stopped` запускает оба процесса
после сбоя/перезапуска Docker. Не удалять volume для обычного рестарта.
Docker-образ не содержит модель и не устанавливает ML-зависимости.

Общий [OpenAPI](../contracts/openapi.yaml) описывает и будущий ML-сервис. Этот процесс
реализует только пять путей: `/health/live`, `/health/ready` и три `/internal/v1/generator/*`.
Его `/openapi.json` содержит только эти пути; `/internal/v1/model` и `/predict` здесь отсутствуют.
Все пути, кроме health, требуют `Authorization: Bearer …`; токен не передавать в URL.
Внутри Docker доступно имя `generator-api:8000`; для локальной проверки порт 8011
привязан только к loopback. Kotlin consumer подключается к той же Docker-сети позже.

## Цикл worker

1. Получить эксклюзивную блокировку на всё время работы. Второй worker завершается
   с ошибкой, не меняя состояние первого.
2. Прочитать checkpoint со всеми очередями, бюджетами, генераторами случайных чисел
   и журналом. При первом старте создать независимые профили и планы клиентов.
3. Каждые `GENERATOR_TICK_SECONDS` вычислить виртуальное время по монотонным часам.
   Время простоя между запусками не догоняется. Один шаг ограничен
   `GENERATOR_MAX_STEP_SECONDS`, чтобы не обрабатывать огромный интервал одним вызовом.
4. Выполнить запланированные задачи до этой границы. Реакции на результаты могут
   запланировать следующие действия. Атомарно сохранить checkpoint.
5. Записать heartbeat; API может отдать только уже сохранённые события.

SIGTERM/SIGINT заканчивает текущий шаг и переводит worker в `paused`. После SIGKILL
доступна последняя целая ревизия. Не подтверждённый сохранением шаг может быть
повторён из прежнего RNG/очереди; наружу он до сохранения не выдавался.
Протокол локального checkpoint подробно описан в [LIVE_STATE.md](../examples/risk/event-driven/LIVE_STATE.md).

| Переменная | Default вне Compose | Значение |
|---|---:|---|
| `GENERATOR_CHECKPOINT` | `runtime/live.json` | Один приватный файл на запуск |
| `GENERATOR_TIME_SCALE` | 60 | Виртуальные секунды на реальную; в Compose 3600 |
| `GENERATOR_TICK_SECONDS` | 1 | Частота попыток продвижения/heartbeat |
| `GENERATOR_MAX_STEP_SECONDS` | 3600 | Максимум виртуальных секунд в одном шаге |
| `GENERATOR_START` | 2026-01-01T00:00:00Z | Начало нового запуска |
| `GENERATOR_DAYS` | 365 | Конечный горизонт нового запуска |
| `GENERATOR_ACCOUNTS` | 50 | Число клиентов нового запуска |
| `GENERATOR_SEED` | 20261024 | Seed только нового запуска |
| `GENERATOR_SERVICE_TOKEN` | нет | Обязательный секрет API, минимум 16 байт; worker его не требует |

На существующем checkpoint настройки создания не пересоздают клиентов. Скорость
можно поменять при рестарте; бизнес-история остаётся той же. Регистрации распределены
по времени, поэтому отсутствие новых событий на отдельном шаге нормально.
`nextEventAt` в статусе помогает видеть ближайшее событие; часы всё равно продвигаются.

Горизонт конечен: после его окончания worker остаётся живым со статусом `finished`,
обновляет heartbeat и сохраняет прежний журнал. Он не сбрасывает курсоры, не запускает
историю заново и не создаёт бесконечные когорты. Продление горизонта — отдельная задача.

## Чтение и доставка

```sh
curl -H "Authorization: Bearer $GENERATOR_SERVICE_TOKEN" \
  http://127.0.0.1:8011/internal/v1/generator/status

# RUN_ID взять из generator.runId в предыдущем ответе.
curl -G -H "Authorization: Bearer $GENERATOR_SERVICE_TOKEN" \
  --data-urlencode "runId=$RUN_ID" --data-urlencode 'afterSequence=0' \
  --data-urlencode 'limit=100' http://127.0.0.1:8011/internal/v1/generator/events
```

Журнал append-only: чтение ничего не удаляет, сервер не хранит единственный общий
«прочитанный курсор». Каждый потребитель хранит собственные `runId` и `nextSequence`.
Номера с 1, `afterSequence=0` читает начало. После обработки порции consumer в одной
транзакции сохраняет изменения и курсор; дедупликация — `(runId,event.id)`.
Этот Kotlin consumer пока не реализован; сервер обеспечивает возможность повторного чтения.

Повтор запроса возвращает те же события на уже заполненных позициях. Если ранее
страница была неполной/пустой, повтор может дополниться новыми событиями; `revision`,
`asOf`, `hasMore` тоже могут измениться. Это не пропуски и не замена старых событий.
Пустая страница сохраняет курсор, `hasMore=false` означает только конец текущей ревизии.
Нельзя увеличивать курсор по времени, количеству попыток или `revision`.

Неизвестный `runId` → 404; курсор за концом журнала → 409; невалидные параметры → 422.
`snapshot?runId=…&expectedRevision=N` → 409, если ревизия уже изменилась. Снапшот
строится из одной прочитанной ревизии. `transactionCoverage=omitted` честно отмечает,
что платежи находятся в журнале; пустой массив transactions не означает отсутствие оплат.

`/health/ready` отвечает за доступность сохранённых данных. Даже если worker упал,
уже сохранённый журнал остаётся доступен. Устаревший heartbeat (> max(10 секунд,
3 интервала worker)) меняет `workerState` с `running` на `error`.
`lastHeartbeatAt`/`lastCheckpointAt` — реальное время; `virtualTime`/`asOf` — виртуальное.
Heartbeat не является состоянием биллинга: при сбое между двумя записями checkpoint
может быть новее heartbeat, но HTTP всегда читает актуальный целый checkpoint.

## Проверка и ограничения

```sh
.venv-generator/bin/pip install -r generator_service/requirements-test.txt
.venv-generator/bin/python -B -m unittest discover -s generator_service/tests -v
```

Тесты запускают настоящие subprocess worker и Uvicorn: автоматические события,
SIGTERM/SIGKILL, запрет второго writer, сохранение RNG/очередей, отсутствие догоняния
простоя, повторные страницы, независимые курсоры, restart API, полное чтение журнала
без пропусков, схема ответов и приватность. ML не нужен для этих проверок.

Состояние пока хранится целиком в JSON (лимит 128 MiB); чтение проверяет и воспроизводит
журнал, поэтому цена чтения/записи растёт. Для большого непрерывного потока потребуется
отдельное хранилище журнала и компактный checkpoint. HTTP polling не является
гарантией exactly-once обработки в БД потребителя. Push/webhook, MinIO и Kotlin consumer
в этом блоке не реализуются. Файл checkpoint нельзя отдавать клиенту или модели.
