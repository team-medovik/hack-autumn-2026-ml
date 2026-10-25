CREATE SCHEMA IF NOT EXISTS generator_sim;
CREATE TABLE IF NOT EXISTS generator_sim.runs (
    run_key TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    revision BIGINT NOT NULL CHECK (revision >= 0),
    virtual_time TIMESTAMPTZ NOT NULL,
    end_at TIMESTAMPTZ NOT NULL,
    event_count BIGINT NOT NULL CHECK (event_count >= 0),
    decision_count BIGINT NOT NULL CHECK (decision_count >= 0),
    state JSONB NOT NULL,
    schema_version TEXT NOT NULL,
    engine_fingerprint TEXT NOT NULL,
    python_version TEXT NOT NULL,
    next_tick_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    clock_config JSONB,
    heartbeat JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (virtual_time <= end_at)
);
ALTER TABLE generator_sim.runs ADD COLUMN IF NOT EXISTS clock_config JSONB;
CREATE TABLE IF NOT EXISTS generator_sim.tasks (
    run_id TEXT NOT NULL REFERENCES generator_sim.runs(run_id),
    queue TEXT NOT NULL CHECK (queue IN ('behavior', 'billing')),
    sequence BIGINT NOT NULL,
    due_at TIMESTAMPTZ NOT NULL,
    phase INTEGER NOT NULL,
    tie_breaker BIGINT NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (run_id, queue, sequence)
);
CREATE INDEX IF NOT EXISTS simulation_tasks_due ON generator_sim.tasks
    (run_id, due_at, queue, phase, tie_breaker, sequence);
CREATE TABLE IF NOT EXISTS generator_sim.events (
    run_id TEXT NOT NULL REFERENCES generator_sim.runs(run_id),
    sequence BIGINT NOT NULL CHECK (sequence > 0),
    event_id TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (run_id, sequence),
    UNIQUE (run_id, event_id)
);
CREATE TABLE IF NOT EXISTS generator_sim.commands (
    run_id TEXT NOT NULL REFERENCES generator_sim.runs(run_id),
    command_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (run_id, command_id)
);
CREATE TABLE IF NOT EXISTS generator_sim.decisions (
    run_id TEXT NOT NULL REFERENCES generator_sim.runs(run_id),
    sequence BIGINT NOT NULL CHECK (sequence > 0),
    payload JSONB NOT NULL,
    PRIMARY KEY (run_id, sequence)
);
CREATE TABLE IF NOT EXISTS generator_sim.cycle_marks (
    run_id TEXT NOT NULL REFERENCES generator_sim.runs(run_id),
    kind TEXT NOT NULL CHECK (kind IN ('planned_cycles', 'noticed_cycles')),
    mark TEXT NOT NULL,
    PRIMARY KEY (run_id, kind, mark)
);
