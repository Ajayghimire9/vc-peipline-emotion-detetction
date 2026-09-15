# EmotionFlow

Text emotion classification with abstention.

EmotionFlow provides a local workflow for classifying labelled emotion text. It supports multiple string labels and keeps the vectorizer and model together, avoiding separate artifacts that can drift out of sync.

## Run locally

Use Python 3.11 or newer in a virtual environment.

```bash
pip install -r requirements-portfolio.txt
python -m emotionflow.pipeline --data examples/texts.csv
```

## Design decisions

The portable workflow accepts text,label CSV data and compares word and character n-gram models with stratified validation.

Predictions below the configured confidence threshold return review status with no assigned label.

The original DVC stage scripts remain as historical experiments. The maintained workflow is in emotionflow; dvc-portfolio.yaml describes its reproducible stage.

## Technology

Python, scikit-learn, pandas, joblib, DVC stage definition, pytest.

## Validation

Run `python -m pytest tests -q` from the repository root. CI runs the maintained test suite and lint checks. Tests use local fixtures or mocks and do not deploy cloud resources.

## Scope and limitations

The happiness, sadness and anger examples are small synthetic fixtures. Emotion labels describe dataset annotations; they do not establish a person’s internal emotional state. No external corpus or model weights are downloaded by the maintained workflow.
