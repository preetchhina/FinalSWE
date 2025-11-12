# Transcriptive – Harnessing AI for Smart Medical Transcription Enhancement

A lightweight, reproducible scaffold for **Team 2 – Project 2** that turns raw clinical notes into useful artifacts:

- **Specialty classification** (TF‑IDF + Logistic Regression / Naive Bayes, or DistilBERT)
- **Entity extraction** (weak‑supervision rules for diagnoses & medications/doses)
- **QA checks** (implausible vitals, missing sections, simple unit sanity)
- **SOAP‑style summarization** (template + extracted entities)

> **Cycle 2 deliverable:** end‑to‑end CLI demo + evaluation reports suitable for the live presentation.

---

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

If you are on macOS + Python 3.9 and hit NumPy/PyTorch wheels issues, try ensuring NumPy < 2.0.

---

## Data Format

Each CSV must provide **`text`** and **`specialty`** columns (with an optional `id`).

```csv
id,text,specialty
1,"Chief complaint: fever and cough for 3 days...", "Pediatrics"
2,"Chest pain on exertion; ECG shows ST changes...", "Cardiology"
```

### Create Stratified Splits

```bash
python src/tools/stratify_split.py \
  --input data/all_notes.csv \
  --text-col text \
  --label-col specialty \
  --outdir data \
  --test-size 0.10 --val-size 0.10 --seed 13
# writes data/train.csv, data/val.csv, data/test.csv + split_summary.txt
```

---

## Training Options

### A) TF‑IDF + Logistic Regression (baseline)

```bash
python src/classify/baselines/train_tfidf_logreg.py \
  --train_csv data/train.csv \
  --val_csv   data/val.csv \
  --test_csv  data/test.csv \
  --text_col  text \
  --label_col specialty \
  --outdir    models/cls_tfidf_logreg
# artifacts: models/cls_tfidf_logreg/{model.joblib,config.json}
# metrics:   reports/metrics/{test_report.json,confusion_matrix.png}
```

### B) TF‑IDF + Multinomial Naive Bayes (baseline)

```bash
python src/classify/baselines/train_tfidf_nb.py \
  --train_csv data/train.csv \
  --val_csv   data/val.csv \
  --test_csv  data/test.csv \
  --text_col  text \
  --label_col specialty \
  --outdir    models/cls_tfidf_nb
# artifacts: models/cls_tfidf_nb/{model.joblib,label_encoder.json}
```

### C) DistilBERT (Transformers)

```bash
python src/classify/train_distilbert.py \
  --train_csv data/train.csv \
  --val_csv   data/val.csv \
  --test_csv  data/test.csv \
  --text_col  text \
  --label_col specialty \
  --outdir    models/cls_distilbert
# artifacts: models/cls_distilbert/{config.json,tokenizer.json,model.safetensors,...}
```

> To keep GitHub small, **do not commit large model files**. Add `models/**` to `.gitignore` or set up **Git LFS** if you must push weights.

---

## Prediction

### Sklearn models (TF‑IDF baselines)

```bash
TEXT="26-year-old with Crohn's disease ... follow-up in 6 weeks."
python src/classify/baselines/predict_sklearn.py models/cls_tfidf_logreg "$TEXT"
# → {"label": "Gastroenterology", "probs": {...}}
```

### DistilBERT

```bash
python src/classify/predict.py models/cls_distilbert "$TEXT"
```

---

## Entity Extraction (Weak NER)

```bash
python -c "from src.weak_ner.extract import extract_entities; \
print(extract_entities(open('data/example_note.txt').read()))"
# → {'diagnoses': [...], 'medications': [{'name':..., 'dose':...}, ...]}
```

---

## QA – Rule‑Based Checks

Run quick quality checks (implausible vitals, missing sections, etc.) and save Markdown or JSON.

```bash
# Direct CLI
python src/qa/report.py --text "$TEXT" --format md --out reports/examples/demo_qa.md
# Or as a module
python -m src.qa.report --text "$TEXT" --format json
```

Output example (Markdown):

```md
### QA Report

- **vital_range** — Temperature out of plausible range
- **section_missing** — Missing section: Medications
```

---

## SOAP‑Style Summaries

```bash
python src/summarize/build.py --text "$TEXT" > reports/examples/soap_summary.txt
# prints a compact SOAP block using extracted entities
```

Example (truncated):

```
=== SOAP Summary ===
Assessment (Dx): Crohn's disease, iron‑deficiency anemia
Medications: adalimumab, azathioprine | doses: 40 mg, 100 mg
Plan: (extract from verbs like start/continue/return — TODO)
```

> The “Plan” line is deliberately conservative in Cycle 2; it will evolve as we expand action‑verb heuristics.

---

## End‑to‑End Demo Script (copy‑paste)

```bash
# 1) Classify
python src/classify/baselines/predict_sklearn.py models/cls_tfidf_logreg "$TEXT"

# 2) QA
python src/qa/report.py --text "$TEXT" --format md --out reports/examples/demo_qa.md
cat reports/examples/demo_qa.md

# 3) Summarize
python src/summarize/build.py --text "$TEXT" > reports/examples/soap_summary.txt
cat reports/examples/soap_summary.txt
```

---

## Reports & Artifacts

- **Classifier metrics:** `reports/metrics/test_report.json`
- **Confusion matrix:** `reports/metrics/confusion_matrix.png`
- **QA examples:** `reports/examples/*.md`
- **Summaries:** `reports/examples/*.txt`

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`**  
  Use the provided CLIs exactly (`python src/...`). Both `src/qa/report.py` and `src/summarize/build.py` self‑bootstrap `sys.path`.
- **Transformers wants the internet** for pretrained downloads. We fine‑tune locally; ensure your `models/cls_distilbert/` has the tokenizer & weights when predicting.

---

## Disclaimer

This is an educational prototype — **not** a medical device. No PHI. Validate outputs clinically before use.
