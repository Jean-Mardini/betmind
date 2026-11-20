"""Gradio app: pick a home and an away team, get win / draw / loss probabilities.

Run `python train.py` first so the models exist in models/, then:

    python app.py
"""

import difflib
import json
import os
import pickle
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

from betmind.features import fixture_features
from betmind.models import TemperatureScaler

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTCOMES = ["Home win", "Draw", "Away win"]
ALIASES = {"man utd": "Man United", "manchester united": "Man United", "manchester city": "Man City",
           "spurs": "Tottenham", "tottenham hotspur": "Tottenham", "wolverhampton": "Wolves",
           "psg": "Paris SG", "paris saint-germain": "Paris SG", "inter milan": "Inter", "bayern munich": "Bayern Munich"}

with open(MODELS / "team_snapshot.pkl", "rb") as f:
    SNAP = pickle.load(f)
TEAMS = sorted(SNAP["teams"])
TEMPS = json.loads((MODELS / "temperatures.json").read_text())
COLS = json.loads((MODELS / "features.json").read_text())["features"]

MODEL_LIST = {}
for name, label in [("logistic_regression", "Logistic Regression"), ("random_forest", "Random Forest")]:
    if (MODELS / f"{name}.joblib").exists():
        MODEL_LIST[label] = (name, joblib.load(MODELS / f"{name}.joblib"))

LSTM = None
if (MODELS / "lstm.keras").exists():
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    import keras

    LSTM = (keras.models.load_model(MODELS / "lstm.keras"), joblib.load(MODELS / "lstm_static_preprocess.joblib"))


def resolve_team(name: str) -> str | None:
    if not name:
        return None
    if name in SNAP["teams"]:
        return name
    key = name.strip().lower()
    if key in ALIASES and ALIASES[key] in SNAP["teams"]:
        return ALIASES[key]
    lower = {t.lower(): t for t in TEAMS}
    if key in lower:
        return lower[key]
    close = difflib.get_close_matches(key, list(lower), n=1, cutoff=0.75)
    return lower[close[0]] if close else None


def scale(name, p):
    return TemperatureScaler._apply(p, TEMPS.get(name, 1.0))


def predict(home_in, away_in):
    home, away = resolve_team(home_in), resolve_team(away_in)
    if home is None or away is None:
        missing = home_in if home is None else away_in
        raise gr.Error(f"Unknown team: {missing}")
    if home == away:
        raise gr.Error("Pick two different teams.")

    x = fixture_features(home, away, SNAP)
    cols = COLS
    rows = []
    for label, (name, model) in MODEL_LIST.items():
        use = cols + ["Division"] if name.startswith("logistic") else cols
        rows.append((label, scale(name, model.predict_proba(x[use]))[0]))
    if LSTM is not None:
        net, prep = LSTM
        static = prep["scaler"].transform(prep["imputer"].transform(x[cols])).astype(np.float32)
        inputs = {"home_seq": SNAP["teams"][home]["seq"][None], "away_seq": SNAP["teams"][away]["seq"][None], "static": static}
        rows.append(("LSTM", scale("lstm", net.predict(inputs, verbose=0))[0]))

    combined = np.mean([p for _, p in rows], axis=0)
    table = pd.DataFrame([[label, *[f"{v:.1%}" for v in p]] for label, p in rows], columns=["Model", *OUTCOMES])
    h, a = SNAP["teams"][home], SNAP["teams"][away]
    info = (f"**{home}** (Elo {h['elo']:.0f}, last 5: {h['form5']:.0f} pts) vs "
            f"**{away}** (Elo {a['elo']:.0f}, last 5: {a['form5']:.0f} pts). "
            f"Based on matches up to {max(h['last_date'], a['last_date']).date()}.")
    return {k: float(v) for k, v in zip(OUTCOMES, combined)}, table, info


with gr.Blocks(title="BetMind") as demo:
    gr.Markdown("# BetMind\nFootball match outcome prediction from Elo ratings and recent form.")
    with gr.Row():
        home_dd = gr.Dropdown(TEAMS, label="Home team", value="Arsenal" if "Arsenal" in TEAMS else None, allow_custom_value=True)
        away_dd = gr.Dropdown(TEAMS, label="Away team", value="Chelsea" if "Chelsea" in TEAMS else None, allow_custom_value=True)
    btn = gr.Button("Predict", variant="primary")
    summary = gr.Markdown()
    with gr.Row():
        label = gr.Label(label="Average of the models", num_top_classes=3)
        table = gr.Dataframe(label="Per model", interactive=False)
    btn.click(predict, [home_dd, away_dd], [label, table, summary])

if __name__ == "__main__":
    demo.launch()
