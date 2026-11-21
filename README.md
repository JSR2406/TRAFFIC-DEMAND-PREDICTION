# Traffic Demand Prediction

A Python regression experiment that predicts a normalized traffic-demand target from location, time, weather, and road features. The script blends LightGBM, XGBoost, CatBoost, and histogram gradient boosting predictions.

## Repository layout

| Path | Purpose |
| --- | --- |
| [solution.py](solution.py) | Feature engineering, model fitting, and submission generation |
| [train.csv](train.csv) | Labeled input with the `demand` target |
| [test.csv](test.csv) | Unlabeled input |
| [sample_submission.csv](sample_submission.csv) | Expected output format |
| [requirements.txt](requirements.txt) | Python libraries used by the script |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Cross-validation caveats and data provenance |

## Reproduce the experiment

Create and activate a Python virtual environment. Run these commands from the repository root.

```bash
python -m pip install -r requirements.txt
python solution.py
```

The script overwrites `submission.csv` in the current directory and prints diagnostic R² values. Interpret those scores in light of the [evaluation caveats](docs/EVALUATION.md).
