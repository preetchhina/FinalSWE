"""
Load any scikit-learn text pipeline (TF-IDF + LogReg or TF-IDF + NB)
and predict a single input string.

Usage:
  python src/classify/baselines/predict_sklearn.py models/cls_tfidf_nb "free text here"
  python src/classify/baselines/predict_sklearn.py models/cls_tfidf_logreg "free text here"
"""

import json, sys
from pathlib import Path
from joblib import load
import numpy as np

def softmax(z):
    # Some sklearn models expose predict_proba directly.
    # If not present, this softmax is a fallback for linear decision_function.
    e = np.exp(z - np.max(z))
    return e / e.sum()

def main():
    if len(sys.argv) < 3:
        print("Usage: predict_sklearn.py <model_dir> \"text to classify\"")
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    text = sys.argv[2]

    # Load pipeline and label map
    pipe = load(model_dir / "model.joblib")
    with open(model_dir / "label_encoder.json") as f:
        classes = json.load(f)["classes"]

    # Prefer predict_proba if available; otherwise approximate from decision_function
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba([text])[0]
    elif hasattr(pipe, "decision_function"):
        scores = pipe.decision_function([text])[0]
        probs = softmax(scores)
    else:
        # As a last resort, make a hard prediction and put prob=1 on top class
        yhat = pipe.predict([text])[0]
        probs = np.zeros(len(classes), dtype=float)
        probs[int(yhat)] = 1.0

    top_idx = int(np.argmax(probs))
    label = classes[top_idx]
    out = {
        "label": label,
        "probs": {cls: float(probs[i]) for i, cls in enumerate(classes)}
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
