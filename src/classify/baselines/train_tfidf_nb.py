#!/usr/bin/env python3
"""
Train a *simple, fast* baseline text classifier:
TF-IDF vectorizer  →  Multinomial Naive Bayes (NB)

Why this exists:
- Gives the team a quick, cheap baseline to compare against DistilBERT.
- Easy to explain in a presentation (“bag of words → TF-IDF → NB”).
- Trains in seconds, runs anywhere.

Inputs:
  CSV files with two columns:
    --text_col   (default: 'text')       e.g. the note/transcription text
    --label_col  (default: 'specialty')  e.g. medical specialty class

Outputs (for reproducibility + reporting):
  - models/cls_tfidf_nb/model.joblib         : scikit-learn pipeline (TF-IDF → NB)
  - models/cls_tfidf_nb/label_encoder.json   : class name ordering for inference
  - reports/metrics/nb_val_report.json       : val macro-F1 + per-class metrics
  - reports/metrics/nb_test_report.json      : test macro-F1 + per-class metrics

Run:
  python src/classify/baselines/train_tfidf_nb.py \
    --train_csv data/train.csv \
    --val_csv   data/val.csv \
    --test_csv  data/test.csv \
    --text_col  text \
    --label_col specialty \
    --outdir    models/cls_tfidf_nb
"""

import argparse, json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


def read_csv(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    """
    Read a CSV and make sure the text column is a string and non-null.

    We do minimal cleaning on purpose (baseline). DistilBERT handles raw text;
    here we rely on TF-IDF to ignore rare tokens and NB smoothing to be robust.
    """
    df = pd.read_csv(path)
    df[text_col] = df[text_col].fillna("").astype(str)
    df[label_col] = df[label_col].astype(str)
    return df


def main():
    # CLI args (keep it simple)
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--test_csv",  required=True)
    ap.add_argument("--text_col",  default="text")
    ap.add_argument("--label_col", default="specialty")
    ap.add_argument("--outdir",    required=True)

    # Sensible defaults that work well on short clinical text
    ap.add_argument("--min_df",    type=int,   default=2,   help="ignore tokens seen < min_df docs")
    ap.add_argument("--ngram_max", type=int,   default=2,   help="use unigrams + bigrams")
    ap.add_argument("--alpha",     type=float, default=0.5, help="NB smoothing; larger = smoother")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load splits
    tr = read_csv(args.train_csv, args.text_col, args.label_col)
    va = read_csv(args.val_csv,   args.text_col, args.label_col)
    te = read_csv(args.test_csv,  args.text_col, args.label_col)

    Xtr, Xva, Xte = tr[args.text_col].tolist(), va[args.text_col].tolist(), te[args.text_col].tolist()

    # Encode string labels → int IDs so sklearn has a stable class order
    le = LabelEncoder()
    ytr = le.fit_transform(tr[args.label_col].tolist())
    yva = le.transform(va[args.label_col].tolist())
    yte = le.transform(te[args.label_col].tolist())
    class_names = list(le.classes_)  # store class list for reports/prediction

    # Build pipeline
    # 1) TF-IDF: turns raw text into sparse numeric features
    #    - min_df removes super-rare tokens (noise)
    #    - (1,2) ngrams let model capture short phrases (“atrial fibrillation”)
    # 2) MultinomialNB: classic, fast text classifier
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=args.min_df, ngram_range=(1, args.ngram_max))),
        ("nb",    MultinomialNB(alpha=args.alpha)),
    ])

    # Train on TRAIN split
    # (Baseline keeps VAL for fair early comparison across models.)
    pipe.fit(Xtr, ytr)

    # Evaluate helper
    def eval_split(X, y, split_name):
        yhat = pipe.predict(X)

        # Use only labels that actually appear in y_true or y_pred for THIS split
        labels_present = sorted(set(y) | set(yhat))
        names_present = [class_names[i] for i in labels_present]

        macro_f1 = f1_score(y, yhat, average="macro")
        report = classification_report(
            y, yhat,
            labels=labels_present,
            target_names=names_present,
            output_dict=True,
            zero_division=0,
        )
        return macro_f1, report


    val_f1, val_report   = eval_split(Xva, yva, "Val")
    test_f1, test_report = eval_split(Xte, yte, "Test")

    # Save artifacts
    dump(pipe, outdir / "model.joblib")  # single file you can load for inference

    with open(outdir / "label_encoder.json", "w") as f:
        json.dump({"classes": class_names}, f, indent=2)

    metrics_dir = Path("reports/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "nb_val_report.json", "w") as f:
        json.dump({"macro_f1": val_f1, "report": val_report}, f, indent=2)
    with open(metrics_dir / "nb_test_report.json", "w") as f:
        json.dump({"macro_f1": test_f1, "report": test_report}, f, indent=2)

    print("[NB][OK] Saved TF-IDF+NaiveBayes to", str(outdir))
    print(f"     Val macro-F1:  {val_f1:.4f}")
    print(f"     Test macro-F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
