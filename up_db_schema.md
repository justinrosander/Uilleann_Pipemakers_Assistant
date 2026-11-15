# up_db PostgreSQL schema

Database: `up_db`  
Schema: `public`  
Host and port come from `DATABASE_URL`. A local install often uses `localhost:5432`.

This is a relational schema. Each committed worksheet becomes one batch, and the measurement rows belong to that batch. It is not a single wide table. The validation report, sanity report, and human-review issues are the only JSON values.

Approving the same sheet again inserts another batch. The schema does not replace an earlier commit of that file and sheet. The instrument row is reused when family, name, pipemaker, and description match.

## instruments

Logical identity of a chanter, regulator, or drone.

| Column | Type | Required | Notes |
|---|---|---|---|
| id | bigint | yes | Primary key |
| instrument_family | text | yes | `chanter`, `regulator`, or `drone` |
| instrument_name | text | yes | Includes the type. Examples: `Chanter`, `Tenor Regulator`, `Baritone Drone`, `Bass Regulator` |
| pipemaker | text | no | Maker name only, such as `A. Maker` |
| bore_type | text | no | Template column `bore type`, such as `Wide` or `Narrow` |
| musical_key | text | no | Template column `key`, such as `D` |
| description | text | no | Longer line, such as `A. Maker Wide Bore D Chanter`. Left as entered |
| units | text | no | `imperial` or `metric` only |
| created_at | timestamptz | yes | Defaults to now |

Unique key: `(instrument_family, instrument_name, pipemaker, description)`.

## instrument_sets

One physical set of pipes: the chanter, regulators, and drones that belong together. A set is identified by name, and usually by pipemaker and musical key.

| Column | Type | Required | Notes |
|---|---|---|---|
| id | bigint | yes | Primary key |
| name | text | yes | Unique. Custodian last name plus maker, for example `Example Set` |
| pipemaker | text | no | |
| musical_key | text | no | |
| notes | text | no | |
| created_at | timestamptz | yes | Defaults to now |

## measurement_batches

One intake of one source sheet. `set_id` places that sheet in an instrument set. When several commits exist for the same sheet, the set keeps the latest one so the bore is not stored twice in an analysis.

| Column | Type | Required | Notes |
|---|---|---|---|
| id | bigint | yes | Primary key |
| source_filename | text | yes | Workbook file name |
| source_sheet | text | yes | Worksheet name |
| instrument_id | bigint | no | Foreign key to `instruments` |
| set_id | bigint | no | Foreign key to `instrument_sets` |
| status | text | yes | Default `pending`. Committed rows use `committed` |
| measurement_method | text | no | Header `Method`, when that label is present |
| collector | text | no | Header `Collector`, when that label is present |
| measurement_date | text | no | Header `Date` as written, such as `3/28/2018` or `YYYY-MM-DD` |
| header_notes | text | no | Header `Notes`. Distinct from row-level `notes` and from `human_notes` |
| validation_report | jsonb | no | |
| sanity_report | jsonb | no | |
| human_notes | text | no | |
| created_at | timestamptz | yes | Defaults to now |
| validated_at | timestamptz | no | |
| committed_at | timestamptz | no | |

## section_oals

One overall length per named section in a batch.

| Column | Type | Required | Notes |
|---|---|---|---|
| id | bigint | yes | Primary key |
| batch_id | bigint | yes | Foreign key to `measurement_batches`. Deleted with the batch |
| section_name | text | yes | Examples: `OAL`, `Upper Bore OAL`, `Reed Seat OAL`, `J1 OAL` |
| oal | numeric | no | |

Unique key: `(batch_id, section_name)`.

## bore_measurements

Bore rows for chanter, regulator, and drone sections. Nominal is stored rounded to 2 decimal places. A text Oval marker such as `through` is stored in `oval_text`, and `oval` is left empty on that row.

| Column | Type | Required | Notes |
|---|---|---|---|
| id | bigint | yes | Primary key |
| batch_id | bigint | yes | Foreign key to `measurement_batches`. Deleted with the batch |
| section_name | text | yes | Default `main`. Also `Upper Bore`, `Lower Bore`, `J1`, and so on |
| row_index | integer | yes | Zero-based order within the section |
| nominal | numeric | no | Rounded to 2 decimal places on insert |
| diameter | numeric | no | May be empty on a notes-only row |
| round_val | numeric | no | Sheet column `Round` |
| oval | numeric | no | Numeric Oval only |
| oval_text | text | no | Non-numeric Oval marker, such as `through` |
| notes | text | no | |

Indexed on `batch_id`.

## tonehole_measurements

Tonehole and keyhole rows.

| Column | Type | Required | Notes |
|---|---|---|---|
| id | bigint | yes | Primary key |
| batch_id | bigint | yes | Foreign key to `measurement_batches`. Deleted with the batch |
| row_index | integer | yes | |
| keyhole | text | no | |
| d_bottom_bell | numeric | no | |
| d_top_reedseat | numeric | no | |
| d_ctr_bell | numeric | no | |
| d_axial | numeric | no | |
| d_xverse | numeric | no | |
| chimney | numeric | no | |
| ch_diam | numeric | no | |
| notes | text | no | |

Indexed on `batch_id`.

## u_bend_measurements

U-bend geometry. Drone U1, U2, and U3 use the shorter set. The folded Bass Upper U-bend may also fill `od` and `notes`. `od` is the maker's estimated roundness from the two OD axes. It is not checked against a formula.

| Column | Type | Required | Notes |
|---|---|---|---|
| id | bigint | yes | Primary key |
| batch_id | bigint | yes | Foreign key to `measurement_batches`. Deleted with the batch |
| section_name | text | yes | Examples: `U1`, `U2`, `U3`, `U-bend` |
| row_index | integer | yes | |
| id_inner | numeric | no | Sheet column `ID` |
| od_x_axis | numeric | no | |
| od_y_axis | numeric | no | |
| od | numeric | no | Estimated round OD |
| wall | numeric | no | |
| diameter_od | numeric | no | |
| radius | numeric | no | |
| length | numeric | no | |
| notes | text | no | |

## slide_brass_tube

Bass drone slide brass tube rows.

| Column | Type | Required | Notes |
|---|---|---|---|
| id | bigint | yes | Primary key |
| batch_id | bigint | yes | Foreign key to `measurement_batches`. Deleted with the batch |
| id_inner | numeric | no | Sheet column `ID` |
| od | numeric | no | |
| wall | numeric | no | |
| length | numeric | no | |

## human_review_queue

Review requests created when automated checks fail.

| Column | Type | Required | Notes |
|---|---|---|---|
| id | bigint | yes | Primary key |
| batch_id | bigint | yes | Foreign key to `measurement_batches`. Deleted with the batch |
| issue_stage | text | yes | Example: `validation_or_sanity` |
| issues | jsonb | yes | List of issue objects |
| status | text | yes | Default `open` |
| human_response | jsonb | no | |
| created_at | timestamptz | yes | Defaults to now |
| resolved_at | timestamptz | no | |

## Relationships

```text
instrument_sets
    ^
    |
instruments
    ^
    |
measurement_batches
    |-- section_oals
    |-- bore_measurements
    |-- tonehole_measurements
    |-- u_bend_measurements
    |-- slide_brass_tube
    \-- human_review_queue
```

Child rows use `ON DELETE CASCADE` from `measurement_batches`, except `measurement_batches.instrument_id`, which does not cascade.
