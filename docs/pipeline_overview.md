# News Classification Pipeline — Verified Repo Description

This document is a **verified, repo‑accurate** description of the end‑to‑end news classification pipeline implemented in this repository. It is structured so it can be copied directly into a report prompt.

## Project goal & scope
This project builds a news classification pipeline that:

1. Loads a labeled development set and an unlabeled evaluation set.
2. Runs detailed exploratory data analysis (EDA), including leakage checks.
3. Trains a linear text classifier with engineered metadata features.
4. Evaluates performance via cross‑validation.
5. Trains a final model on all development data and generates predictions for the evaluation set, producing a submission file.

## Data inputs and schema

### Default input files
Default inputs are defined in `configs/default.json` and point to:

- `data/raw/development.csv`
- `data/raw/evaluation.csv`

### Required columns (main pipeline)
The main pipeline (via `newsclf.io`) **expects lowercase columns**:

- `development.csv`: `id`, `source`, `title`, `article`, `page_rank`, `timestamp`, `label`
- `evaluation.csv`: `id`, `source`, `title`, `article`, `page_rank`, `timestamp`

The IO layer standardizes `Id` or `ID` to `id` and throws a clear error if any required columns are missing.

### Required columns (EDA)
The standalone EDA script **expects a fixed, strict schema** with the **capitalized `Id` column** and exact column ordering:

- `development.csv`: `Id`, `source`, `title`, `article`, `page_rank`, `timestamp`, `label`
- `evaluation.csv`: `Id`, `source`, `title`, `article`, `page_rank`, `timestamp`

## EDA (Exploratory Data Analysis)
The EDA script (`src/newsclf/eda.py`) provides a report‑style diagnostic of both splits and optionally saves plots.

### Schema enforcement & dataset profiling
- Enforces **exact column order** for dev/eval (schema mismatch throws).
- Prints row/column counts, dtypes, memory usage, and a fingerprint hash.

### Missingness & placeholder detection
- Counts NA, empty strings, placeholder values (`\N` and timestamp placeholders).
- Reports missingness for text fields by label (dev only).

### ID and duplicate checks
- Missing/duplicate ID counts, with ID min/max or cardinality.
- Fully duplicated rows, duplicate `(title, article)` pairs, and duplicate titles.
- Label‑conditional duplicate detection:
  - Exact duplicates (hash of title+article).
  - Optional **near‑duplicate** check with normalized text (lowercase + punctuation strip).

### Text and categorical diagnostics
- Text length stats (chars/words), URL/HTML/entity/non‑ASCII prevalence, and samples.
- Source distribution and `page_rank` consistency by source.
- Timestamp quality (placeholder checks, parseability, year/hour distributions).

### Label distribution and imbalance
- Per‑label counts and share.
- Majority‑class macro‑F1 baseline estimate.
- Per‑label text length summaries and time‑by‑label summaries.

### Cross‑split drift & leakage heuristics
- Checks overlap of IDs between dev/eval.
- Source overlap and unseen eval sources.
- Jensen‑Shannon divergence on `page_rank`, `timestamp.year`, and binned word lengths.
- Exact overlap on title/article hashes to flag potential leakage/reprints.

### Optional model‑aware diagnostics
If provided with out‑of‑fold predictions, EDA can:
- Print true vs. predicted label counts.
- Compute per‑class PR curves when score columns are available (prefers `score_*`, `prob_*`, or `logit_*` prefixes).
- Plot score distributions for weak labels.

### Optional plots
With `--plots`, EDA saves summary histograms and bar charts to `reports/eda/`.

## Leakage prevention & deduplication (main pipeline)
The unified entry point (`src/newsclf/main.py`) **cleans the development set** prior to modeling by:

1. **Dropping dev rows overlapping eval** on `(title, article)` using canonicalization:
   - HTML unescape → strip tags → replace URLs → Unicode normalize → strip accents → whitespace collapse → lowercase.
2. **Dropping cross‑label duplicates** in the dev set using the same canonicalization.

These steps run both during cross‑validation and final training in `main.py`.

> Note: Legacy scripts (`newsclf.cv`, `newsclf.train_final`, `newsclf.predict`) do **not** perform these leakage‑prevention steps.

## Feature engineering & preprocessing
The model uses a `ColumnTransformer` with text, categorical, numeric, and time features (`newsclf.model` + `newsclf.features`).

### Text pipeline
- `AdPrefixStripper`: removes advertisement/sponsored prefixes from title/article.
- `TextJoiner`: repeats the title (`title_repeat`) and concatenates with article; appends a missing‑article token when appropriate.
- `TextCleaner`: HTML unescape, strip tags, replace URLs, Unicode normalize, collapse whitespace.
- TF‑IDF `FeatureUnion`:
  - Word n‑grams (`ngram_range`, `min_df`, `max_features`).
  - Character n‑grams (`char_wb`, 3–5 grams, capped feature size).
- Optional **title‑only char TF‑IDF** when `text.title_char=true`.

### Text pattern metadata
`TextPatternFeatures` computes:
- Character and word lengths (title & article).
- Presence of ad prefixes, URLs, HTML entities, and non‑ASCII characters.
- Scaled with `StandardScaler(with_mean=False)`.

### Source categorical feature
- `SourceBinner` groups rare sources into `__OTHER__` and missing into `__MISSING__`.
- One‑hot encoding with `OneHotEncoder(handle_unknown="ignore")`.

### Numeric & timestamp features
- `page_rank`: numeric coercion + scaling.
- `timestamp`: parsed to datetime and expanded to:
  - `year`, `month`, `day_of_year`, `day_of_week`
  - cyclic encodings for hour, day‑of‑week, month, and day‑of‑year
- A separate **timestamp missingness indicator** is added.
- A **missing‑article indicator** is added.

All features are concatenated into a single sparse design matrix.

## Modeling choices
Supported classifiers (`model.type`):

- `linearsvc` (default): linear SVM for high‑dimensional sparse text.
- `logreg`: logistic regression (liblinear).
- `ridge`: ridge classifier baseline.

Class weighting:

- `class_weight` supports `"balanced"` or `None`.
- `class_weight_power` adjusts the balanced weights by exponentiation.

## Cross‑validation stage (Stage 1)
Implemented in `src/newsclf/main.py`:

- Uses `StratifiedKFold` with k folds and a fixed seed.
- For each fold:
  - Builds the full pipeline with configured hyperparameters.
  - Fits on the train fold and predicts the validation fold.
  - Computes macro/micro/weighted F1 and per‑class metrics.
  - Aggregates confusion matrix and true vs predicted counts.

Outputs saved to `cfg.paths.cv_out_dir` (default `reports/cv`):

- `folds.csv` (metrics per fold)
- `per_class_mean.csv`
- `confusion_matrix.csv`
- `oof_true_pred_counts.csv`
- `cv_summary.json`
- Plots: `folds_macro_f1.png`, `per_class_f1.png`, `confusion_matrix.png`

## Final training & evaluation prediction (Stage 2)
Also handled in `src/newsclf/main.py`:

- Re‑applies leakage and duplicate cleaning on the full dev set.
- Trains on all dev data with the same pipeline.
- Saves a serialized model (`model.pkl`).
- Predicts labels for `evaluation.csv`.
- Writes submission file with required schema: `Id, Predicted`.
- Saves prediction counts and a distribution plot.

Outputs are saved under the **testing output directory**, which is:

- `--run_dir/testing` when `--run_dir` is provided, or
- the parent directory of `paths.submission_out` (default `models/`).

## Entry points and configuration

### Unified entry point (recommended)
`src/newsclf/main.py` is the **most complete pipeline**, supporting leakage cleaning and full reporting.

Key flags:

- `--set section.key=value` to override config values.
- `--run_dir` to place outputs under a custom run folder.
- `--skip_cv` / `--skip_test` to skip stages.
- `--cache_dir` to enable joblib transform caching.

### Legacy modular scripts
Documented in `howto.txt` and still available:

- `python -m newsclf.cv`
- `python -m newsclf.train_final`
- `python -m newsclf.predict`

> These legacy scripts are simpler and **do not** include leakage cleaning or all config‑driven features.

## Default configuration highlights (`configs/default.json`)

- CV: 5 folds, seed 42
- Text: 1–2 grams, min_df=5, max_features=100k, title_repeat=1 (no extra repeats), title‑char features enabled
- Model: linearsvc, C=4.0, max_iter=10000, class_weight=balanced
- Source binning: min_count=1 (rare sources retained)
- Outputs: `reports/cv`, `models/model.pkl`, `models/submission.csv`

## End‑to‑end pipeline narrative (one‑paragraph summary)
The pipeline begins with EDA to validate schema, quantify missingness, study label imbalance, detect duplicates, and compare development vs. evaluation distributions. Before training, the unified `main.py` cleans the development set to prevent leakage by removing any `(title, article)` pairs overlapping the evaluation set and by removing cross‑label duplicates. Features are built from cleaned text using TF‑IDF word and char n‑grams, supplemented with text‑pattern metadata, source one‑hot encodings, numeric `page_rank`, timestamp‑derived cyclic features, and missingness indicators. A linear classifier (default Linear SVM) is trained with stratified k‑fold cross‑validation, producing fold‑level metrics, per‑class scores, and confusion matrices. Finally, the model is retrained on the full cleaned development set and used to predict the evaluation set, producing a submission CSV and saved model artifact.
