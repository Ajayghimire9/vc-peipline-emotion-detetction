"""Local CSV training with train-only TF-IDF and a portable fitted pipeline."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


def prepare(frame):
    if not {"text", "label"}.issubset(frame.columns) or frame.empty:
        raise ValueError("CSV requires nonempty text and label columns")
    if frame[["text", "label"]].isna().any().any():
        raise ValueError("Null text or labels are not supported")
    result = frame[["text", "label"]].astype(str).copy()
    result["text"] = (
        result["text"].str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    )
    if result["text"].str.len().eq(0).any():
        raise ValueError("Blank text is not supported")
    if result.groupby("text")["label"].nunique().gt(1).any():
        raise ValueError("Identical text has conflicting labels")
    result = result.drop_duplicates("text").reset_index(drop=True)
    if result["label"].nunique() < 2 or result["label"].value_counts().min() < 6:
        raise ValueError(
            "At least two labels and six distinct examples per label are required"
        )
    return result


def train(source, output):
    frame = prepare(pd.read_csv(source))
    train_df, test_df = train_test_split(
        frame, test_size=0.25, stratify=frame.label, random_state=42
    )
    candidates = {}
    for name, analyzer, ngrams in [
        ("word", "word", (1, 2)),
        ("character", "char_wb", (3, 5)),
    ]:
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer=analyzer, ngram_range=ngrams, max_features=20000
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced", max_iter=500, random_state=42
                    ),
                ),
            ]
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, train_df.text, train_df.label, cv=cv, scoring="f1_macro"
        )
        candidates[name] = (float(scores.mean()), model)
    name = max(candidates, key=lambda key: candidates[key][0])
    model = candidates[name][1].fit(train_df.text, train_df.label)
    predictions = model.predict(test_df.text)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "model.joblib"
    joblib.dump(model, path)
    report = {
        "selected": name,
        "validation_macro_f1": {k: v[0] for k, v in candidates.items()},
        "test_macro_f1": float(f1_score(test_df.label, predictions, average="macro")),
        "test_report": classification_report(
            test_df.label, predictions, output_dict=True, zero_division=0
        ),
        "labels": model.classes_.tolist(),
        "confusion_matrix": confusion_matrix(
            test_df.label, predictions, labels=model.classes_
        ).tolist(),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def predict(model_path, texts, minimum_confidence=0.65):
    if (
        not 0 <= minimum_confidence <= 1
        or not texts
        or any(not str(t).strip() for t in texts)
    ):
        raise ValueError("Provide nonblank texts and a confidence threshold in [0,1]")
    path = Path(model_path)
    manifest = json.loads(path.with_name("metrics.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["model_sha256"]:
        raise ValueError("Artifact integrity check failed")
    model = joblib.load(path)  # Only load artifacts from a trusted training run.
    probabilities = model.predict_proba(texts)
    return [
        {
            "label": str(model.classes_[p.argmax()])
            if p.max() >= minimum_confidence
            else None,
            "confidence": float(p.max()),
            "status": "accepted" if p.max() >= minimum_confidence else "review",
        }
        for p in probabilities
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="examples/texts.csv")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output), indent=2))
