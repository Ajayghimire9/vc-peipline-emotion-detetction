import pandas as pd
import pytest

from emotionflow.pipeline import predict, prepare, train


def test_training_and_artifact_roundtrip(tmp_path):
    report = train("examples/texts.csv", tmp_path)
    assert report["train_rows"] > report["test_rows"] > 0
    result = predict(tmp_path / "model.joblib", ["please help with this request"])
    assert result[0]["status"] in {"review", "accepted"}
    (tmp_path / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        predict(tmp_path / "model.joblib", ["request"])


def test_conflicting_duplicates_are_rejected():
    frame = pd.DataFrame({"text": ["SAME text", "same  text"], "label": ["a", "b"]})
    with pytest.raises(ValueError, match="conflicting"):
        prepare(frame)
