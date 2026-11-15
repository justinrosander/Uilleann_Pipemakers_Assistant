-- Measurement database schema. Safe to run more than once.

CREATE TABLE IF NOT EXISTS instruments (
    id                BIGSERIAL PRIMARY KEY,
    instrument_family TEXT NOT NULL,
    instrument_name   TEXT NOT NULL,
    pipemaker         TEXT,
    bore_type         TEXT,
    musical_key       TEXT,
    description       TEXT,
    units             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT instruments_units_check
        CHECK (units IS NULL OR units IN ('imperial', 'metric')),
    CONSTRAINT instruments_identity_key
        UNIQUE (instrument_family, instrument_name, pipemaker, description)
);

CREATE TABLE IF NOT EXISTS instrument_sets (
    id           BIGSERIAL PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    pipemaker    TEXT,
    musical_key  TEXT,
    notes        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS measurement_batches (
    id                  BIGSERIAL PRIMARY KEY,
    source_filename     TEXT NOT NULL,
    source_sheet        TEXT NOT NULL,
    instrument_id       BIGINT REFERENCES instruments (id),
    set_id              BIGINT REFERENCES instrument_sets (id),
    status              TEXT NOT NULL DEFAULT 'pending',
    measurement_method  TEXT,
    collector           TEXT,
    measurement_date    TEXT,
    header_notes        TEXT,
    validation_report   JSONB,
    sanity_report       JSONB,
    human_notes         TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    validated_at        TIMESTAMPTZ,
    committed_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS measurement_batches_set_idx
    ON measurement_batches (set_id);

CREATE TABLE IF NOT EXISTS section_oals (
    id           BIGSERIAL PRIMARY KEY,
    batch_id     BIGINT NOT NULL REFERENCES measurement_batches (id) ON DELETE CASCADE,
    section_name TEXT NOT NULL,
    oal          NUMERIC,
    UNIQUE (batch_id, section_name)
);

CREATE TABLE IF NOT EXISTS bore_measurements (
    id           BIGSERIAL PRIMARY KEY,
    batch_id     BIGINT NOT NULL REFERENCES measurement_batches (id) ON DELETE CASCADE,
    section_name TEXT NOT NULL DEFAULT 'main',
    row_index    INTEGER NOT NULL,
    nominal      NUMERIC,
    diameter     NUMERIC,
    round_val    NUMERIC,
    oval         NUMERIC,
    oval_text    TEXT,
    notes        TEXT
);

CREATE INDEX IF NOT EXISTS bore_measurements_batch_idx
    ON bore_measurements (batch_id);

CREATE TABLE IF NOT EXISTS tonehole_measurements (
    id              BIGSERIAL PRIMARY KEY,
    batch_id        BIGINT NOT NULL REFERENCES measurement_batches (id) ON DELETE CASCADE,
    row_index       INTEGER NOT NULL,
    keyhole         TEXT,
    d_bottom_bell   NUMERIC,
    d_top_reedseat  NUMERIC,
    d_ctr_bell      NUMERIC,
    d_axial         NUMERIC,
    d_xverse        NUMERIC,
    chimney         NUMERIC,
    ch_diam         NUMERIC,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS tonehole_measurements_batch_idx
    ON tonehole_measurements (batch_id);

CREATE TABLE IF NOT EXISTS u_bend_measurements (
    id           BIGSERIAL PRIMARY KEY,
    batch_id     BIGINT NOT NULL REFERENCES measurement_batches (id) ON DELETE CASCADE,
    section_name TEXT NOT NULL,
    row_index    INTEGER NOT NULL,
    id_inner     NUMERIC,
    od_x_axis    NUMERIC,
    od_y_axis    NUMERIC,
    od           NUMERIC,
    wall         NUMERIC,
    diameter_od  NUMERIC,
    radius       NUMERIC,
    length       NUMERIC,
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS slide_brass_tube (
    id        BIGSERIAL PRIMARY KEY,
    batch_id  BIGINT NOT NULL REFERENCES measurement_batches (id) ON DELETE CASCADE,
    id_inner  NUMERIC,
    od        NUMERIC,
    wall      NUMERIC,
    length    NUMERIC
);

CREATE TABLE IF NOT EXISTS human_review_queue (
    id             BIGSERIAL PRIMARY KEY,
    batch_id       BIGINT NOT NULL REFERENCES measurement_batches (id) ON DELETE CASCADE,
    issue_stage    TEXT NOT NULL,
    issues         JSONB NOT NULL,
    status         TEXT NOT NULL DEFAULT 'open',
    human_response JSONB,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at    TIMESTAMPTZ
);
