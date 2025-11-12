# -----------------------------------------------------------------------------
# Baseline: TF-IDF + Logistic Regression (multiclass)
#
# Usage:
# python src/classify/baselines/train_tfidf_logreg.py \
#  --train_csv data/train.csv \
#  --val_csv   data/val.csv \
#  --test_csv  data/test.csv \
#  --text_col text \
#  --label_col specialty \
#  --outdir models/cls_tfidf_logreg
#
# What this does:
#   • Reads your stratified splits (train/val/test)
#   • Vectorizes text with TF-IDF (uni+bi-grams)
#   • Trains a Logistic Regression classifier (one-vs-rest)
#   • Saves: model.joblib, config.json, and val/test JSON reports (incl. confusion matrix)
#
# Why we add it:
#   • FAST to train on CPU (seconds) — great demo baseline
#   • Interpretable (we can later inspect top n-grams)
#   • Handles class imbalance via class_weight="balanced"
# -----------------------------------------------------------------------------

import argparse, json, pathlib, joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def load_xy(csv_path: str, text_col: str, label_col: str):
    """
    Return X (list[str]) and y (list[str]) from a CSV.
    Casting to str avoids None/NaN sneaking into the vectorizer.
    """
    df = pd.read_csv(csv_path)
    X = df[text_col].astype(str).tolist()
    y = df[label_col].astype(str).tolist()
    return X, y


def evaluate(pipe: Pipeline, X, y_ids, classes, outdir: pathlib.Path, split_name: str):
    """
    Predict on a split, write JSON report + confusion matrix.
    y_ids are numeric labels (0..K-1); 'classes' are class names by index.
    """
    y_pred = pipe.predict(X)

    # classification_report wants numeric y but we can pass readable names for rows
    rep = classification_report(
        y_true=y_ids,
        y_pred=y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,  # don't crash if a class has 0 support
    )

    cm = confusion_matrix(y_ids, y_pred).tolist()
    payload = {
        "split": split_name,
        "report": rep,
        "confusion_matrix": cm,
        "labels": classes,
    }
    (outdir / f"{split_name}_report.json").write_text(json.dumps(payload, indent=2))

    return rep["macro avg"]["f1-score"]


def main():
    ap = argparse.ArgumentParser()
    # Defaults match the structure you described (data/csvs/*)
    ap.add_argument("--train_csv", default="data/csvs/train.csv")
    ap.add_argument("--val_csv",   default="data/csvs/val.csv")
    ap.add_argument("--test_csv",  default="data/csvs/test.csv")
    # Column names used in your repo (change if your CSV uses different names)
    ap.add_argument("--text_col",  default="text")
    ap.add_argument("--label_col", default="specialty")
    # Where to save this baseline’s artifacts
    ap.add_argument("--outdir",    default="models/cls_tfidf_logreg")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load splits
    Xtr, ytr = load_xy(args.train_csv, args.text_col, args.label_col)
    Xva, yva = load_xy(args.val_csv,   args.text_col, args.label_col)
    Xte, yte = load_xy(args.test_csv,  args.text_col, args.label_col)

    # 2) Stable label mapping (sorted for reproducibility)
    classes = sorted(set(ytr) | set(yva) | set(yte))          # class names
    label2id = {lab: i for i, lab in enumerate(classes)}      # str -> int
    id2label = {i: lab for lab, i in label2id.items()}        # int -> str

    # Map string labels -> numeric IDs for sklearn
    ytr_i = [label2id[y] for y in ytr]
    yva_i = [label2id[y] for y in yva]
    yte_i = [label2id[y] for y in yte]

    # 3) Build the pipeline: TF-IDF (1–2 grams) + Logistic Regression
    pipe = Pipeline([
        # TF-IDF: cap features + log TF; drops rare/ubiquitous n-grams
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),     # unigrams + bigrams catch short medical phrases
            min_df=3,               # drop super-rare n-grams
            max_df=0.85,            # drop ubiquitous n-grams
            max_features=60000,
            sublinear_tf=True,
            stop_words="english"    # remove English stopwords
        )),

        # Logistic Regression: liblinear OvR is fast on sparse text
        ("clf", LogisticRegression(
            solver="liblinear",
            multi_class="ovr",
            class_weight="balanced",
            C=1.0,
            tol=1e-3,
            max_iter=1000,
            n_jobs=-1,
            verbose=1    # shows progress per class so you can see it working
        ))
    ])

    # 4) Train
    pipe.fit(Xtr, ytr_i)

    # 5) Evaluate + write JSONs
    f1_val = evaluate(pipe, Xva, yva_i, classes, outdir, "val")
    f1_tst = evaluate(pipe, Xte, yte_i, classes, outdir, "test")

    # 6) Save model + minimal config (mirrors HF style so your codebase is consistent)
    joblib.dump(pipe, outdir / "model.joblib")
    meta = {
        "id2label": {str(i): lab for i, lab in id2label.items()},
        "label2id": label2id,
        "val_macro_f1": f1_val,
        "test_macro_f1": f1_tst
    }
    (outdir / "config.json").write_text(json.dumps(meta, indent=2))

    print(f"[OK] Saved TF-IDF+LogReg to {outdir}")
    print(f"     Val macro-F1:  {f1_val:.4f}")
    print(f"     Test macro-F1: {f1_tst:.4f}")


if __name__ == "__main__":
    main()