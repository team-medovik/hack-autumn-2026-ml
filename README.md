# Aeza VPS/VDS — генератор событий и модель риска

В репозитории работают два независимых сервиса:

- **Генератор:** PostgreSQL хранит очередь будущих действий, состояние/RNG и журнал; worker выполняет задачи, FastAPI отдаёт сохранённые события и снимки.
- **ML:** отдельный FastAPI загружает готовый CatBoost и оценивает риск потери VPS. Генерацию и обучение он не запускает.

[Kotlin API](https://github.com/team-medovik/hack-autumn-2026-api) читает события по курсору, сохраняет inbox и проекцию, затем передаёт опубликованную историю в ML. Он же объединяет вероятность с алгоритмом: 30% расхождения, максимум ±20 баллов. При отказе ML сохраняется алгоритм.

Полная связка подготовлена в парных ветках `feature/live-stack-integration` обоих репозиториев. Наличие кода здесь не означает, что API уже обновлён на сервере: нужна соответствующая версия Kotlin и включённые флаги. Публичные данные не содержат скрытых намерений клиентов, seed или состояния RNG.

## Быстрый запуск Python-сервисов

Требуются Docker Compose с `up --wait`, существующая Docker-сеть `backend` и заранее подготовленные секреты. Скрипт не создаёт токены и не запускает/не меняет API.

```bash
cp .env.example .env.live
# Заполнить пустые значения секретов в .env.live и установить права 600.
chmod 600 .env.live
# backend — существующая сеть контейнера API на этом же Docker-хосте.
bash scripts/live-stack.sh validate "$PWD/.env.live"
bash scripts/live-stack.sh up "$PWD/.env.live"
bash scripts/live-stack.sh check "$PWD/.env.live"
bash scripts/live-stack.sh status "$PWD/.env.live"
```

В `.env.example` для **нового** запуска указано `GENERATOR_PAYMENT_TERMS=hour,day,month`.
В существующем сохранённом запуске периоды услуг не переписываются; без явной настройки
worker сохраняет совместимый default `month`. Суточные цены остаются синтетическими.

Полный интерфейс: `validate`, `up`, `status`, `check`, `logs`, `stop`. `stop` сохраняет БД; обычный перезапуск не должен удалять volume. Сервисы разделены на стабильные Compose-проекты `aeza-generator` и `aeza-ml`. Изменение имени проекта означает другое пространство volumes — не использовать это для «починки» существующего запуска.

Для изолированного стенда можно явно задать `LIVE_STACK_NAMESPACE` и
`LIVE_STACK_NETWORK`, а порты — через env-файл. Эти значения надо повторять при
каждом вызове; другой namespace создаёт другие volumes.

Для перехода со старого Compose задайте в shell `LIVE_STACK_GENERATOR_PROJECT`
равным **фактическому прежнему project**: тогда сохраняется тот же PG volume и run.
`LIVE_STACK_ML_PROJECT` отдельно переопределяет проект ML; оба override имеют
приоритет над namespace и должны передаваться при каждом вызове. Перед переходом
остановите старые worker/API, сделайте backup и сохраните прежние настройки часов
и пароль БД. [Пошаговый переход и проверка volume](generator_service/README.md#переход-с-существующего-postgresql-запуска).
Не запускайте вторую PostgreSQL с тем же volume; `down -v` не нужен.

- [compose.generator.production.yaml](compose.generator.production.yaml): PostgreSQL, worker и generator-api; БД в приватной сети, HTTP API также в `backend`.
- [compose.ml.production.yaml](compose.ml.production.yaml): только модель, адрес `ml-service:8000`, readiness и локальный диагностический порт.
- [Подробности генератора](generator_service/README.md), [подробности inference](inference_service/README.md).

## Подключение Kotlin API

Секреты API должны совпадать с соответствующими секретами Python-сервисов. Для текущего деплоя API настройки хранятся в его GitHub Secret `ENV_PRODUCTION`; не заменять существующие datasource/auth поля.

```dotenv
APP_GENERATOR_ENABLED=true
APP_GENERATOR_USE_FOR_DASHBOARD=true
APP_GENERATOR_SOURCE_ID=aeza-live
APP_GENERATOR_BASE_URL=http://generator-api:8000
APP_GENERATOR_SERVICE_TOKEN=<значение GENERATOR_SERVICE_TOKEN>
APP_ML_ENABLED=true
APP_ML_BASE_URL=http://ml-service:8000
APP_ML_SERVICE_TOKEN=<значение ML_SERVICE_TOKEN>
APP_ML_MODEL_VERSION=event-driven-all-terms-v1-234422ba1357
```

Эти имена разрешаются внутри общей Docker-сети. `localhost:8011/8012` из контейнера API указывает на сам API, а не Python-сервисы. Оба механизма по умолчанию выключены; до публикации первой live-проекции данные не подменяются произвольным JSON.

После нового опубликованного снимка Kotlin использует новую ревизию/время для прогноза; кеш прошлой ревизии не подходит. SSE `entity-changed` с `entity=SNAPSHOT` сообщает фронтенду, что нужно перечитать данные. Генератор не отправляет прогноз в API напрямую.

## Текущая модель

- Версия: `event-driven-all-terms-v1-234422ba1357`, CatBoost с Logloss, 36 признаков.
- Периоды: `hour/day/month`; суточная цена синтетическая.
- Прогноз до срока оплаты, в пределах 7 суток. Цель: нет продления/подтверждённой оплаченной замены к сроку + 5 суток.
- Артефакт [inference_service/artifacts/model.cbm](inference_service/artifacts/model.cbm) и manifest включены в Git и Docker-образ; обучение при запуске не требуется.
- [Последний отчёт по всем периодам](ml/ALL_TERMS_TRAINING.md). На месячном test новая модель немного хуже старой по Logloss/Brier. Синтетические проверки не доказывают качество на настоящих клиентах.

`prolong_wait`, просроченные/удалённые/заменённые услуги и нефинансовые ограничения не получают ML-прогноз. Kotlin применяет правила; для `prolong_wait` — F3+F4. Отсутствующий прогноз нельзя заменять нулём.

## Контракт и эксплуатация

[Внутренний OpenAPI](contracts/openapi.yaml) описывает независимые API. Health доступен без токена; внутренние маршруты требуют Bearer service token.

| Сервис | Основные маршруты |
|---|---|
| generator-api | `/health/ready`, `/internal/v1/generator/status`, `/internal/v1/generator/events`, `/internal/v1/generator/snapshot` |
| ml-service | `/health/ready`, `/internal/v1/model`, `/internal/v1/renewal-risk/predict` |

Readiness генератора означает, что сохранённое состояние читается. Для проверки живой генерации нужны heartbeat, `workerState` и продвижение виртуального времени. Повторное чтение не удаляет события; курсоры и дедупликация принадлежат Kotlin.

Сохранять всю PostgreSQL-БД генератора, включая очередь/состояние/журнал. Не выполнять `down -v` при обновлении. RUN_KEY и настройки часов относятся к сохранённому запуску: новый seed не пересоздаёт его, несовместимые часы отклоняются. Горизонт конечен; `finished` не запускает новую симуляцию автоматически.

Текущая Kotlin-интеграция передаёт полный префикс: до 100000 событий, 32 MiB, 1000 запрошенных услуг, timeout 2 s по умолчанию. Превышение/неполная история приводит к fallback. MinIO dataset registry и автоматическое архивирование пока не реализованы. Это не бесконечный production-поток без ограничения размера истории.

Автодеплой модели уже добавлен в `develop` отдельным PR #6; его workflow и скрипт
сохранены. `scripts/live-stack.sh` запускается оператором и не изменяет GitHub Secrets.
Не запускайте ручной `up` одновременно с автодеплоем того же проекта `aeza-ml`: у
launcher нет deploy-lock, и он заменяет ML локально собранным образом. Для стенда
используйте отдельные namespace/network/порты; production ML обновляйте одним путём.

## Деплой ML в production

После обновления `main` workflow `ML basic checks` проверяет коммит. Если CI
успешен, `Deploy ML` собирает Docker-образ именно этого коммита на сервере и
запускает `ml-service` через `compose.ml.production.yaml`. Скрипт ждёт readiness,
проверяет версию и SHA модели по `manifest.json` и при ошибке пытается вернуть
предыдущий релиз. При обычном процессе релиза это срабатывает после слияния
`develop` в `main`. Деплой генератора и API ведётся отдельно.

Для первого запуска нужен Linux self-hosted runner с метками `self-hosted`,
`linux`, `deploy` в этом репозитории, установленными Docker Engine, Compose v2,
Git, Bash, Python 3 и `flock`. Runner должен иметь доступ к Docker, каталогу
`/home/kinoko/hack-autumn-2026/ml` и созданной на сервере сети `backend`.
Зарегистрируйте в GitHub Actions secret `ML_SERVICE_TOKEN` (32–256 символов
`A–Z`, `a–z`, `0–9`, `_` или `-`); его значение должно совпадать с
`APP_ML_SERVICE_TOKEN` у Kotlin API. Каталог релизов и секретный `runtime.env`
создаются на сервере и не попадают в Git.

Текущий релиз: `/home/kinoko/hack-autumn-2026/ml/current`. Проверка на сервере:

```sh
curl --fail http://127.0.0.1:8012/health/ready
```

Для ручного повторного деплоя запустите `ML basic checks` через
`workflow_dispatch` на актуальном `main`. Предыдущие релизы сохраняются в
`/home/kinoko/hack-autumn-2026/ml/releases` для ручного отката.

## Проверка связки — 25.10.2026

Реальный worker → PostgreSQL → HTTP → Kotlin → CatBoost проверен без ручного
заполнения inbox: 9549 → 9564 событий, совпадение проекций, сохранение runId/курсоров
после перезапуска, повторное чтение без повторных списаний. Проверены SSE новой
публикации, свежие прогнозы и возврат к алгоритму при остановке ML с последующим
автоматическим восстановлением. HYBRID получен для часа, суток и месяца.

Прошли 4 проверки выбора периодов, 6 проверок launcher (переход на прежний project,
медленный tick и остановка прогресса), 12 существующих live/HTTP-проверок и валидация
обоих Compose. Локальные тесты и логи: `runtime/live-stack-tests/`; сквозной отчёт
хранится в API checkout, `build/live-stack-check/`. Новые тесты не включены в Git
согласно согласованному процессу; это результаты локальной проверки, а не CI.

## Разработка и данные

Команды выполняются из корня ML-репозитория, независимо от checkout API:

```bash
python3.13 -m venv ml/.venv
ml/.venv/bin/python -m pip install -r ml/requirements-lock.txt
ml/.venv/bin/python -B -m unittest discover -s examples/risk/event-driven/tests
ml/.venv/bin/python -B -m unittest discover -s ml/tests -p test_event_driven_dataset.py
```

CI проверяет существующие генератор, подготовку признаков и внутренний контракт без переобучения. Новые локальные интеграционные проверки перед PR исключаются из публикации по согласованному процессу; не считать их частью CI, если исходников нет в ветке.

`ml/runs/` с большими датасетами и результатами не входит в Git. Сервисный артефакт хранится отдельно в `inference_service/artifacts/`. Исторические эксперименты и воспроизведение описаны в [ml/EVENT_DRIVEN_TRAINING.md](ml/EVENT_DRIVEN_TRAINING.md) и [ml/ALL_TERMS_TRAINING.md](ml/ALL_TERMS_TRAINING.md).

## История переноса — 25 октября 2026

Источник: ветка `feature/risk-contract-alignment` API, коммит `c5ff60b`.
История извлечена средствами `git fast-export` / `git fast-import` по путям
`ml`, `examples/risk`, `src/test/resources/risk/data.json`,
`outputs/event-driven-logloss-20261025` и `AGENTS.md`.
Авторы, даты и сообщения сохранены; SHA изменены из-за фильтрации деревьев.
История API не переписывалась.

| Коммит API | Коммит ML |
|---|---|
| `a722e49` | `902194d` |
| `c30a264` | `c1af2c2` |
| `0212094` | `76a7f41` |
| `8d4e434` | `e1d2454` |
| `4942dac` | `b58a500` |
| `a05eaaa` | `b45f425` |
| `c5ff60b` | `206ec98` |

Ссылки на старые SHA API в исторических отчётах нужно читать через эту таблицу.
Копии генераторов в API оставлены для совместимости его примеров;
дальнейшую разработку ML и генератора ведём здесь.

## Проверка первоначального переноса (исторические результаты)

- Извлечённое Git-дерево совпало с выбранными путями исходного коммита;
  метаданные всех 7 коммитов проверены автоматически.
- 63 файла ML (включая локальные выборки и модели, без окружения, кешей и
  обновлённого README) совпали побайтно.
- В новом checkout прошли 59 event-driven ML-тестов и 125 тестов генератора.
- Полный исторический набор ML-тестов прерван вручную на длительной подготовке
  старого датасета: его полное прохождение не подтверждено этим переносом.
- На момент первоначального переноса проверка Kotlin в API была заблокирована отсутствием JDK 25; окружение содержит
  JDK 22/23. Это не ошибка теста ML и не подтверждение корректности Kotlin-изменений.

Полный пример строки: [VPS 3000180, 30 января 2026](ml/reports/sample-3000180-20260130.json).
Исходная локальная папка `ml` сохранена резервной копией внутри API checkout:
`.git/ml-migration-backup-20261025/ml`. Она не публикуется в GitHub.
