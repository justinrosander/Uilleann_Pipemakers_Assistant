# Uilleann Pipemaker's AI Assistant
Version: 1.0.0
## Agentic AI RAG Workflow for Uilleann Pipe Measurement 
## Intake, Machine Learning powered analytics, 
## and Neural Network framework for tonehole optimization.

This is Uilleann Pipemaker's AI Assistant--an Agentic AI RAG Workflow for Uilleann Pipe Measurement Intake, Machine Learning powered analytics, and Neural Network framework for tonehole optimization.

LangGraph orchestrator that intakes bore/tonehole spreadsheets against `UP_measurement_template.xlsx`, runs deterministic validation and sanity checks, pauses for human review on failure, and commits approved data to PostgreSQL (`up_db`).

The EDA agent, bore optimizer, and tonehole model run when you ask for them, from the command line or from the chat.

## Directions

Set up .env and config.py appropriately with Grok API and PostgreSQL db setup info.  Both are optional but extremely helpful.

`UP_measurement_template.xlsx` is a required file.  It does not contain production data, so do not use to produce real instruments, or you will waste time and materials.  However, do use this file as a template to log your measurements of real instruments, as your data intake will be much faster for approval.

## Chat orchestrator

`py main.py --chat` opens a conversation that can run intake, EDA, the bore optimizer, and the tonehole model. The command-line flags for those same steps stay available. The chat uses `XAI_API_KEY`.

When a check fails, the chat can wait while you edit the workbook. It continues only after you say the file is saved. Before an EDA report, it asks whether you want extra observations in that report. Those notes are written into the HTML or PDF for that run.

```powershell
py main.py --chat
```

## Setup

From the project directory:

```powershell
cd path\to\project
py -m pip install -r requirements.txt
copy .env.example .env
```

### Environment

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | PostgreSQL URL (for example `postgresql://user@localhost:5432/up_db`) |
| `TEMPLATE_PATH` | Canonical validation template (default `UP_measurement_template.xlsx`) |
| `CHECKPOINT_PATH` | LangGraph SQLite checkpoint DB (default `data/langgraph_checkpoints.sqlite`) |
| `XAI_API_KEY` | Optional; core validation does not require an LLM |

If `DATABASE_URL` has no password, the app can read it from the system keyring. Set `DATABASE_KEYRING_SERVICE` and `DATABASE_KEYRING_USER` in `.env`. The setup script writes the password into `.env` instead:

```powershell
py scripts/setup_postgres.py
```

That script creates the application database when it is missing and applies `db/schema.sql`. An existing database is kept. Git ignores `.env`.

### PostgreSQL and spreadsheet-only use

PostgreSQL (`up_db`) is required to commit intake batches, create or list instrument sets, and run agents with `--set`.

A user who stays on workbooks does not need PostgreSQL for these steps:

- Preprocess, which writes Excel under `data/processed`
- EDA with `--source`, which writes HTML or PDF under `data/eda`
- Bore optimizer with `--source`, which writes Excel and HTML under `data/bore_optimizer`
- Tonehole model with `--source`, which writes HTML or PDF under `data/tonehole_model`

LangGraph checkpoints use SQLite under `data/langgraph_checkpoints.sqlite` and do not need PostgreSQL.

## Validation template

Incoming workbooks are checked against rules derived from **`UP_measurement_template.xlsx`** (sheet names, headers, OAL labels, section layout). Those rules live in `schema/template_schema.py`.

Provide spreadsheets that follow that template.

**Supported intake formats:** `.xlsx`, `.xlsm`, `.xltx`, `.xltm`, `.ods` (OpenDocument via `odfpy`).

PDF / handwritten intake is not implemented yet and will route to human review.

## Preprocess non-standard files
`UP_measurement_template.xlsx` is a required file.  It does not contain production data, so do not use to produce real instruments, or you will waste time and materials.
Use this before intake when a file is not already laid out like `UP_measurement_template.xlsx`. Accepted sources are `.csv`, `.txt`, `.tsv`, `.xlsx`, `.xlsm`, `.ods`, and `.pdf`.

```powershell
py main.py --preprocess --source "data\raw\measurements.csv"
```

What happens:

1. A copy of the source is saved in `data/raw`.
2. The file is scanned for structured tables and for unstructured text.
3. You confirm shared fields such as pipemaker, last name, key, and optional header Method/Collector/Date/Notes.
4. Instrument title and optional voice stay per sheet from `UP_measurement_template.xlsx` cell A1 (for example `Chanter`, `Tenor Regulator`, `Bass Drone`). They are not entered once and stamped onto every sheet.
5. For each table you can map columns onto a template sheet, skip the table, or stop and edit the workbook by hand. Mapping a second source column onto a target that already has data appends those values in empty rows below; it does not overwrite.
6. Unstructured text can be placed in the header Method, Collector, Date, or Notes fields, or left out.
7. A copy of the template, with example measurement rows cleared, is saved in `data/processed`. The file name is `{LastName}_{Key}_{InstrumentTitles}.xlsx`, for example `Smith_D_Chanter.xlsx` or `Smith_D_Chanter_Tenor_Regulator.xlsx` when more than one sheet is kept. Sheets that received no data are removed from that workbook. A `.mapping.txt` file next to it records the choices.

Accepted mappings are stored in `data/preprocess_memory/mappings.jsonl`. The next preprocess run retrieves similar examples and uses them for suggestions. If `XAI_API_KEY` is set, those examples are also sent to Grok as context. You still accept, change, or skip every mapping. With no API key, matching uses column names plus the saved examples.

The processed workbook is the file to pass to normal intake:

```powershell
py main.py --source "data\processed\Smith_D_Chanter.xlsx" --all-sheets --thread-id run-processed
```

## EDA framework

`ML EDA Problem Definition.md` is the business context and the analytical objectives. The agent reports bore profiles, round versus oval reconstructions, chanter and regulator comparisons, regressions through a 5th-order polynomial, and correlations at `|r| >= 0.7`. It also reports the derived features added in that brief: minor, major, and area-equivalent diameters, ovality, local taper rate, linear-cone residuals, and the multivariate checks (correlation heatmap, principal components, and bore segments).

```powershell
py main.py --eda
py main.py --eda --source "data\processed\Smith_D_Chanter.xlsx" --format html
py main.py --eda --source "data\processed\Smith_D_Chanter.xlsx" --format pdf
py main.py --eda --source "data\processed\Smith_D_Chanter.xlsx" --format both
py main.py --eda --set "Example Set" --format html
```

A database set groups the chanter, regulators, and drones that belong together. Create one from committed sheets that share a pipemaker and key. The latest commit of each sheet is attached, so an older duplicate is not analyzed twice.

```powershell
py main.py --create-set --pipemaker "A. Maker" --key D
py main.py --list-sets
```

Pass `--set "Example Set"` on a later intake run to file that committed sheet into the same set. The set name is the custodian's last name plus the maker, which records the chain of custody. The pipemaker stored on the set stays the maker name you entered, for example `A. Maker`.

## Bore optimizer

The bore optimizer turns round-probe and oval-probe stations into an area-equivalent design. At each oval-probe depth it intercepts the minor diameter from the round-probe curve. At each round-probe depth it intercepts the major diameter from the oval-probe curve. The design diameter is D_eq = √(D_min × D_maj). Results go to a workbook and an HTML figure under `data/bore_optimizer`. The database is not updated.

```powershell
py main.py --optimize-bore --set "Example Set"
py main.py --optimize-bore --source "data\processed\Smith_D_Chanter.xlsx"
```

## Tonehole model

The tonehole agent studies hole placement and chimney size against the bore. On a sparse set it trains a one-layer ANN, a two-layer MLP, and a 1D CNN, and it reports leave-one-out error. Benade's open-hole cutoff, Nederveen's chimney corrections, and the Lefebvre and Scavone short-hole fit are separate sections. The report is HTML, PDF, or both, under `data/tonehole_model`. The database is not updated.

```powershell
py main.py --tonehole-model --set "Example Set" --format html
py main.py --tonehole-model --set "Example Set" --format pdf
py main.py --tonehole-model --source "data\processed\Smith_D_Chanter.xlsx" --format both
```

Without `--source`, `--eda` prints the plan only. With `--source` or `--set`, it visualizes each bore, compares the chanter with the regulators, fits linear and quadratic curves between the round and oval measurements, plots tonehole placement, and checks relationships between tonehole position and bore diameter. Univariate summaries and correlations at `|r| >= 0.7` cover the numeric bore and tonehole fields. Depth is the mean of Round and Oval when both exist. Reports are written under `data/eda/`.

## Pipeline

`intake → validate → sanity → (human review on failure) → commit approval → DB commit → post-commit verify`

Hallucination tolerance is **zero**: validation/sanity are rule-based Python, not LLM guesses.

## CLI

There is **no positional filename argument**. Always pass the workbook with `--source`.

### Flag reference

| Flag | Meaning |
|------|---------|
| `--source PATH` | Workbook to ingest (`.xlsx` / `.xlsm` / `.ods`). Defaults to `UP_measurement_template.xlsx` if omitted. Relative paths are resolved from the project root. |
| `--sheet "Name"` | Worksheet to ingest (e.g. `"Tenor Drone Bore"`). If omitted, the first known template sheet is used. |
| `--all-sheets` | Process every template sheet in workbook order. Each sheet has its own check, commit approval, and post-commit confirmation. Reject or declining a commit skips that sheet and continues. |
| `--list-sheets` | Print worksheet names in `--source` and exit. |
| `--thread-id ID` | LangGraph thread id for a run. **Reuse the same id** when resuming. Default: `up-intake-1`. |
| `--resume` | Continue a paused run (after human review / commit / post-commit gates). |
| `--decision approve\|reject\|correct` | Decision when resuming a failed validation/sanity review. |
| `--commit-approved` | Approve writing the batch to PostgreSQL. |
| `--post-commit-verified` | Confirm you verified the committed rows in `up_db`. |
| `--notes "..."` | Optional human notes attached to a resume. |
| `--auto-approve` | Smoke-test only: skip human gates and commit if checks pass. |

### Typical workflow

```powershell
# From the project directory

# 1) Inspect the workbook (.xlsx or .ods)
py main.py --source "data\raw\example.xlsx" --list-sheets
py main.py --source "data\raw\example.ods" --list-sheets

# 2) Start intake for one sheet (human gates ON)
py main.py --source "data\raw\example.xlsx" --sheet "Tenor Drone Bore" --thread-id run-1
py main.py --source "data\raw\example.ods" --sheet "Tenor Drone Bore" --thread-id run-ods-1

# Every template sheet, one approval at a time (same --thread-id throughout)
py main.py --source "data\processed\Smith_D_Chanter.xlsx" --all-sheets --thread-id run-all
```

When the graph pauses, use the **same** `--thread-id`:

```powershell
# If validation/sanity failed:
py main.py --resume --thread-id run-1 --decision approve
#   accept the data despite the issues
py main.py --resume --thread-id run-1 --decision correct
#   you edited the source file; re-read it and run checks again
py main.py --resume --thread-id run-1 --decision reject

# If checks passed and it asks to commit (either flag works):
py main.py --resume --thread-id run-1 --commit-approved --notes "Looks good"
py main.py --resume --thread-id run-1 --decision approve

# After commit, confirm DB review:
py main.py --resume --thread-id run-1 --post-commit-verified
```

Thread state is stored in `data/langgraph_checkpoints.sqlite`, so `--resume` works across separate PowerShell commands. If you see “No checkpoint found”, start a new intake with that `--thread-id` first. Old in-memory runs (before this fix) cannot be resumed — restart intake.

### Using the built-in template as the source

Useful for demos or reloading known-good sample data:

```powershell
py main.py --list-sheets
py main.py --sheet "Chanter Bore" --thread-id demo-chanter
# (--source defaults to UP_measurement_template.xlsx)
```

### Smoke test (local only)

Bypasses human interrupt gates. Do **not** use for real production intake.

```powershell
py main.py --source "data\raw\example.xlsx" --sheet "Chanter Bore" --auto-approve
```

### Help

```powershell
py main.py --help
```

## PostgreSQL tables

These tables are used when intake commits to the database or when agents load a named set. Spreadsheet-only preprocess and `--source` analysis do not need them.

Created in `up_db`:

- `instruments`
- `measurement_batches`
- `section_oals`
- `bore_measurements`
- `tonehole_measurements`
- `u_bend_measurements`
- `slide_brass_tube`
- `human_review_queue`

## Project layout

| Path | Role |
|------|------|
| `main.py` | CLI entrypoint |
| `agents/` | Intake, validation, sanity, human review, DB commit, orchestrator |
| `schema/template_schema.py` | Template sheet/column expectations |
| `db/` | PostgreSQL connection and writers |
| `UP_measurement_template.xlsx` | Canonical measurement / validation template |
| `config.py` | Loads `.env` and resolves `TEMPLATE_PATH` / `DATABASE_URL` |
