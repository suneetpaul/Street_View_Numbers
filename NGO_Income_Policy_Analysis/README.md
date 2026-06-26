# Income Bracket Estimator

A Streamlit app that estimates the probability of an individual earning
above $50,000/year, built for DeltaSquare NGO's social policy analysis.

## What's in this repo

```
app.py                 Streamlit app — loads model.pkl and serves the form
model.pkl              Trained sklearn pipeline (preprocessing + logistic regression)
model_metadata.pkl     Form dropdown options, numeric ranges, and test metrics
train_model.py         Script used to (re)train model.pkl from raw data
requirements.txt       Python dependencies for Streamlit Cloud
data/who_data.csv      Raw training data (only needed if you retrain)
```

The app does **not** retrain on every run — it loads the already-trained
`model.pkl` at startup, so deployment is fast and doesn't need the dataset.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub (public or private).
2. Go to https://share.streamlit.io and sign in with your GitHub account.
3. Click **"New app"**, select this repo, branch `main`, and set the main
   file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` and runs
   `app.py` — `model.pkl` is loaded directly since it's already in the repo.

## Retraining the model

If you update the dataset and want to retrain:

```bash
python train_model.py
```

This regenerates `model.pkl` and `model_metadata.pkl`. Commit both updated
files back to the repo (and redeploy, or Streamlit Cloud will auto-redeploy
on push if you've connected it that way).

## Notes

- `model.pkl` and `model_metadata.pkl` are small (a few KB) and safe to
  commit directly — no Git LFS needed.
- The `scikit-learn` version in `requirements.txt` is pinned to match the
  version used to train `model.pkl`. If you retrain with a different
  scikit-learn version, update this pin to match.
- This tool produces a statistical estimate based on historical patterns —
  it is intended for policy triage and group-level insight, not individual
  eligibility decisions.
