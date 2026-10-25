# Постоянный событийный генератор

Генератор работает независимо от ML. Worker исполняет будущие действия клиентов
и биллинга; FastAPI отдаёт сохранённые события и текущий публичный снимок.
Kotlin читает страницы по курсору и применяет их через собственный PostgreSQL inbox.
HTTP-запросы не запускают генерацию.

## Запуск

Для совместного запуска с готовой моделью используйте из корня репозитория:

```sh
# Заполнить .env.live существующими секретами оператора по образцу .env.example.
scripts/live-stack.sh validate /absolute/path/.env.live
scripts/live-stack.sh up /absolute/path/.env.live
scripts/live-stack.sh check /absolute/path/.env.live
```

Скрипт использует `compose.generator.production.yaml` и `compose.ml.production.yaml`.
Проекты по умолчанию — `aeza-generator` и `aeza-ml`; PostgreSQL сохраняется в volume
`aeza-generator_generator-postgres`. Уже существующая сеть `backend` нужна для API.
Только `generator-api` и `ml-service` входят в неё: БД и worker остаются во внутренней
сети генератора. API скрипт не запускает и его конфигурацию не меняет.

`validate` проверяет Compose и секреты без вывода их значений. `up` сначала собирает
образы, затем запускает сервисы и проверяет модель/heartbeat/движение времени.
`check` также проверяет доступность ML по Docker DNS из generator-api; это проверка
сети, а не вызов модели генератором. `status`, `logs`, `stop` показывают состояние,
последние логи или останавливают процессы. Команд удаления/reset/генерации токенов нет.

Для отдельного проверочного окружения можно передать **в shell**, не в dotenv:
`LIVE_STACK_NAMESPACE=aeza-check LIVE_STACK_NETWORK=aeza-check-network`.
Сеть предварительно создаёт оператор; скрипт не создаёт её. Порты задаются в env-файле
через `GENERATOR_HTTP_PORT`/`ML_HTTP_PORT` (по умолчанию 8011/8012, только loopback).
Смена namespace означает другие проекты и другой volume, а не перенос старого запуска.
Не подключайте два окружения с одинаковыми service alias к одной сети.
Для изолированной проверки задайте также собственные `GENERATOR_IMAGE_TAG` и
`ML_IMAGE_TAG`, чтобы не перезаписать тег образа рабочего окружения.

### Переход с существующего PostgreSQL-запуска

Старый `docker compose -f compose.generator.yaml up` обычно создавал проект
`hack-autumn-2026-ml`, а не `aeza-generator`. Не запускайте новый `up` с defaults
поверх такого стенда: он создаст другой volume и новый run. Сначала найдите
контейнер **действующей** БД и проверьте его Compose label и подключённый volume:

```bash
set -euo pipefail
# Используйте прежние -p/--env-file, если старый запуск выполнялся с ними.
old_db=$(docker compose -f compose.generator.yaml ps -q generator-db)
test -n "$old_db"
LIVE_STACK_GENERATOR_PROJECT=$(docker inspect --format '{{ index .Config.Labels "com.docker.compose.project" }}' "$old_db")
test -n "$LIVE_STACK_GENERATOR_PROJECT"
export LIVE_STACK_GENERATOR_PROJECT
old_volume=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql/data"}}{{.Name}}{{end}}{{end}}' "$old_db")
test "$old_volume" = "${LIVE_STACK_GENERATOR_PROJECT}_generator-postgres"
```

Имя volume должно быть `${LIVE_STACK_GENERATOR_PROJECT}_generator-postgres`.
Если это bind mount, внешний volume или другой DSN/пользователь/имя БД, этот путь
не подходит: сначала нужен отдельный Compose-конфиг для фактического хранилища;
не запускайте новый worker на пустой БД. Не подключайте один PG volume к двум
контейнерам PostgreSQL одновременно.

В `.env.live` перенесите прежние пароль PostgreSQL, service token, `RUN_KEY` и все
три настройки часов. Если старый пароль — `generator-local` или содержит символы,
не подходящие для URI, сначала отдельно смените пароль роли БД и настройку старого
стека: новая переменная окружения сама пароль существующей БД не меняет.
Для ручной смены оператор может открыть `docker exec -it "$old_db" psql -U generator -d generator`
и выполнить `\password generator`: psql запросит пароль без включения его в текст
команды/history. Это отдельная операция с согласованным обновлением старых клиентов;
скрипт её не выполняет и секретов не создаёт.
Проверьте, что старые и новые образы используют совместимые Python/движок.

```bash
bash scripts/live-stack.sh validate /absolute/path/.env.live
# Останавливаем обе старые службы (включая все реплики worker), БД оставляем работать.
docker compose -p "$LIVE_STACK_GENERATOR_PROJECT" --env-file /absolute/path/.env.live -f compose.generator.yaml stop generator-worker generator-api
# Сохранить исходный runId/revision/event_count для проверки после перехода.
docker exec "$old_db" psql -U generator -d generator -c 'SELECT run_key,run_id,revision,event_count FROM generator_sim.runs;'
umask 077
docker exec "$old_db" pg_dump -U generator -d generator -Fc > generator-before-live.dump
bash scripts/live-stack.sh up /absolute/path/.env.live
```

Новое описание использует тот же project, имена служб и volume, поэтому Compose
обновляет существующие контейнеры, а не поднимает параллельную БД. После `up`
проверьте прежний runId и неуменьшившиеся revision/eventCount через status API;
сохраните прежний sourceId и курсоры Kotlin. При несовместимости не удаляйте данные:
вернитесь к прежним образам/настройкам после остановки новых worker/API.

`LIVE_STACK_GENERATOR_PROJECT` и `LIVE_STACK_ML_PROJECT` задаются **в shell** и
имеют приоритет над `LIVE_STACK_NAMESPACE`; передавайте их при каждом вызове.
ML должен иметь отдельный project (по умолчанию `aeza-ml`). `check` читает интервал
запущенного worker и ждёт продвижения времени до `max(10, 3 × tickSeconds + 5)` секунд;
остановившийся, завершённый или откатившийся запуск не считается работающим.

В `.env.example` включено `GENERATOR_PAYMENT_TERMS=hour,day,month`: новый запуск
содержит все три периода. Маленькая случайная популяция не гарантирует каждую категорию.
Без настройки периодов старое поведение сохранено: только `month`. Периоды, seed и
размер популяции относятся только к созданию запуска и не переписывают существующий.
Для сохранённого запуска надо сохранить также прежние настройки часов: новый production
Compose по умолчанию использует скорость 60, а старый `compose.generator.yaml` — 3600.

Ниже остаётся самостоятельный локальный запуск генератора (без ML и без подключения
к `backend`); для production предпочтителен описанный выше строгий конфиг.

Задайте `GENERATOR_SERVICE_TOKEN` (минимум 16 байт) и `GENERATOR_POSTGRES_PASSWORD`:

```sh
docker compose -f compose.generator.yaml up -d --build
# Две реплики, один общий запуск и темп часов:
docker compose -f compose.generator.yaml up -d --scale generator-worker=2
```

Compose запускает PostgreSQL 16, worker и FastAPI. БД сохраняется в named volume
`generator-postgres`. Worker применяет миграцию и создаёт запуск только при его
отсутствии. API не выполняет DDL и не создаёт запуск; до инициализации readiness=503.
Не удаляйте volume при обычном рестарте.

Для внешнего PostgreSQL задайте `GENERATOR_DATABASE_URL`, затем используйте
`docker compose -f compose.generator.yaml up -d --build --no-deps generator-worker generator-api`.
Генератору нужна собственная БД/роль; worker создаёт схему `generator_sim`.
HTTP-приложению после миграции достаточно SELECT: при отдельном развёртывании
можно передать ему read-only DSN. Compose использует одну роль для обоих процессов.

## Что хранится в PostgreSQL

| Таблица в `generator_sim` | Назначение |
|---|---|
| `runs` | runId, ревизия, виртуальное время, счётчики, настройки часов, heartbeat, приватное состояние JSONB |
| `tasks` | Будущие действия и таймеры с исходными ключами порядка |
| `events` | Публичный журнал, уникальные `(run_id,sequence)` и `(run_id,event_id)` |
| `commands` | Идентификаторы команд для идемпотентности |
| `decisions` | Приватная история решений синтетических клиентов |
| `cycle_marks` | Исторические отметки платёжных циклов |

JSONB содержит аккаунты, услуги, бюджеты, потребности, RNG и служебные поля.
Журнал, решения и очереди не дублируются в нём. Текущие объекты/служебные карты
продолжают расти с портфелем; это не полностью нормализованная модель данных.

Каждый tick в одной транзакции:

1. Блокирует строку запуска `FOR UPDATE SKIP LOCKED`, проверяет `next_tick_at`.
2. Восстанавливает приватное состояние/RNG и ожидающие задачи в локальный heap.
3. Выполняет задачи до границы шага, завершая каждый одинаковый виртуальный момент
   целиком: биллинг, наблюдение результата, реакции клиентов.
4. Сохраняет задачи, состояние/RNG, новые события, решения, время и ревизию вместе.

Внутри запуска сохраняется последовательный порядок: VPS клиента делят баланс.
Занятый запуск пропускается другим worker, но отдельные задачи не пропускаются.
Долговременный источник очереди — БД; `heapq` только исполняет её в памяти.
Номера событий выделяются транзакционно, без глобального BIGSERIAL с пропусками.

До commit сбой откатывает весь шаг; после commit процесс продолжает с сохранённой
границы. При ошибке подключения retry: 1, 2, 4 … 30 секунд, сброс после успеха.
Несовместимость версии или настроек часов не подавляется. Соединения/SQL/блокировки
имеют ограниченные таймауты. Ошибка не разрешает пропустить задачу и двигаться дальше.

## Время и конфигурация

Шаг виртуального времени: `min(tickSeconds*timeScale,maxStepSeconds)`.
Общее расписание хранится в БД; два worker не ускоряют часы вдвое. Простой не
догоняется; под нагрузкой симуляция может идти медленнее заданной скорости.
Горизонт конечен: после него worker обновляет heartbeat `finished`, историю не сбрасывает.

| Переменная | Default вне Compose | Назначение |
|---|---|---|
| `GENERATOR_DATABASE_URL` | пусто | PostgreSQL URL; без него файловый backend |
| `GENERATOR_RUN_KEY` | `default` | Стабильное имя запуска для всех его реплик/API |
| `GENERATOR_TIME_SCALE` | 60 | Скорость; локальный Compose: 3600, production: 60 |
| `GENERATOR_TICK_SECONDS` | 1 | Интервал попыток шага |
| `GENERATOR_MAX_STEP_SECONDS` | 3600 | Максимум виртуальных секунд за шаг |
| `GENERATOR_START` | `2026-01-01T00:00:00Z` | Начало нового запуска |
| `GENERATOR_DAYS` | 365 | Горизонт нового запуска |
| `GENERATOR_ACCOUNTS` | 50 | Клиентов в новом запуске |
| `GENERATOR_SEED` | 20261024 | Seed нового запуска |
| `GENERATOR_PAYMENT_TERMS` | `month` | Периоды через запятую для нового запуска: hour/day/month; `.env.example` включает все три |
| `GENERATOR_SERVICE_TOKEN` | нет | Секрет HTTP, минимум 16 байт |
| `GENERATOR_CHECKPOINT` | `runtime/live.json` | Только файловый backend |

Seed и размер популяции не пересоздают существующий запуск. Настройки часов
фиксируются в БД при первом worker; разные настройки реплик отклоняются.
Простая смена env не меняет скорость сохранённого запуска. Другой RUN_KEY создаёт
независимые runId и историю, поэтому Kotlin требует отдельной привязки источника.

## Импорт существующего checkpoint

Остановите файловый worker, сохраните checkpoint и старый volume. Импорт выполните
до первого PG worker: иначе он автоматически создаст новый запуск под тем же ключом.
Установите `generator_service/requirements.txt`, задайте целевой DSN и выполните
в окружении с той же версией движка и тем же Python major/minor:

```sh
python -m generator_service.postgres --checkpoint /absolute/path/live.json --run-key default
```

Импорт атомарный и сохраняет runId, revision, event IDs, номера, очереди и RNG.
Существующий RUN_KEY/runId никогда не перезаписывается. Несовместимая версия
checkpoint отклоняется. Затем переключите FastAPI и worker на этот PG/RUN_KEY.
Kotlin сохраняет прежние sourceId и курсоры. После появления новых PG-событий
возвращать источник к старому файлу нельзя: это откат уже опубликованной истории.

## HTTP и Kotlin

[OpenAPI](../contracts/openapi.yaml) не менялся: health и три `/internal/v1/generator/*`
пути status/events/snapshot. Все пути, кроме health, требуют Bearer token.
Локальный порт Compose 8011 привязан к loopback; внутри сети — `generator-api:8000`.

`GET /internal/v1/generator/events?runId=…&afterSequence=0&limit=100` читает страницу
индексированного журнала. Метаданные и события читаются согласованно в REPEATABLE READ.
Status/events не загружают приватный JSONB или всю историю. Snapshot использует
текущее состояние без replay. Чтение ничего не удаляет; каждый consumer хранит
свой курсор. Пустая страница позже может дополниться новыми событиями.
`hasMore=false` означает конец текущего журнала, а не завершение симуляции.
Недоступность БД → 503, неизвестный runId → 404, неверный курсор/ревизия → 409.

Kotlin consumer и вызов ML реализованы в API-ветке `feature/live-stack-integration`;
наличие кода в ветке не означает его production-деплой. Историческая реализация
consumer — [PR API #34](https://github.com/team-medovik/hack-autumn-2026-api/pull/34).
Его inbox/проекция/курсоры независимы от внутренних таблиц генератора. Kotlin передаёт
опубликованную историю отдельному ML-сервису; генератор этого не делает. Архетипы,
RNG, бюджеты и скрытые планы не входят в публичные ответы.

## MinIO и ограничения

Рабочее состояние в PostgreSQL: оно должно коммититься вместе с задачами и событиями.
Запись объекта MinIO не входит в SQL-транзакцию. MinIO подходит для архивных снимков,
дампов БД и датасетов. Автоматический экспорт/восстановление из MinIO в этом блоке
не реализованы. Резервировать нужно всю схему `generator_sim`, а не один JSONB:
без задач и журнала восстановление неполное.

Файловый backend сохранён для offline-сценариев. Перенос не делает горизонт
бесконечным, не добавляет webhook и не интегрирует ML. Пока каждый tick загружает
весь хвост ожидающих задач; накопленный журнал при этом не загружается.
Очистка/архивирование журнала — отдельная политика с учётом курсоров потребителей.
Результаты проверок: [VERIFICATION.md](VERIFICATION.md).
