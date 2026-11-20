"""Train and evaluate all BetMind models.

    python train.py              # logistic regression, random forest, LSTM
    python train.py --no-lstm    # skip the LSTM (faster)

Writes the models to models/ and the test metrics to results/.
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from betmind.data import LABELS, load_matches, split_mask
from betmind.features import SEQ_COLS, SEQ_LEN, build_features, build_sequences, feature_columns, team_snapshot
from betmind.models import TemperatureScaler, logistic_regression, lstm, random_forest

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SEED = 0


def metrics(y, p):
    onehot = np.eye(3)[y]
    return {
        "accuracy": round(float(accuracy_score(y, p.argmax(1))), 4),
        "macro_f1": round(float(f1_score(y, p.argmax(1), average="macro")), 4),
        "log_loss": round(float(log_loss(y, p, labels=[0, 1, 2])), 4),
        "brier": round(float(np.mean(np.sum((p - onehot) ** 2, axis=1))), 4),
        "predicted_draw_share": round(float(np.mean(p.argmax(1) == 1)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    np.random.seed(SEED)

    t0 = time.time()
    df = load_matches()
    m, t = build_features(df)
    print(f"{len(m)} matches, features built in {time.time() - t0:.0f}s")

    tr, va, te = split_mask(m)
    te = te & m[["odds_p_home", "odds_p_draw", "odds_p_away"]].notna().all(axis=1)  # same test rows for every model
    y_tr, y_va, y_te = m.loc[tr, "y"].to_numpy(), m.loc[va, "y"].to_numpy(), m.loc[te, "y"].to_numpy()
    cols = feature_columns(m)
    cols_odds = feature_columns(m, with_odds=True)
    print(f"train {tr.sum()}  val {va.sum()}  test {te.sum()}  ({len(cols)} features)")

    results, probs = {}, {}

    # Baselines
    results["always_home"] = metrics(y_te, np.tile(np.bincount(y_tr, minlength=3) / len(y_tr), (len(y_te), 1)))
    results["always_home"]["accuracy"] = round(float(np.mean(y_te == 0)), 4)
    p_book = m.loc[te, ["odds_p_home", "odds_p_draw", "odds_p_away"]].to_numpy()
    results["bookmaker_odds"] = metrics(y_te, p_book)

    temps = {}
    sk_models = {
        "logistic_regression": (logistic_regression(cols), cols + ["Division"]),
        "random_forest": (random_forest(cols, seed=SEED), cols),
        "logistic_regression_with_odds": (logistic_regression(cols_odds), cols_odds + ["Division"]),
    }
    for name, (model, use_cols) in sk_models.items():
        t1 = time.time()
        model.fit(m.loc[tr, use_cols], y_tr)
        scaler = TemperatureScaler().fit(model.predict_proba(m.loc[va, use_cols]), y_va)
        p = scaler.transform(model.predict_proba(m.loc[te, use_cols]))
        probs[name] = p
        temps[name] = scaler.t
        results[name] = metrics(y_te, p)
        joblib.dump(model, MODELS / f"{name}.joblib", compress=3)
        print(f"{name:32s} acc {results[name]['accuracy']:.4f}  log loss {results[name]['log_loss']:.4f}  ({time.time() - t1:.0f}s)")

    if not args.no_lstm:
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        import keras

        keras.utils.set_random_seed(SEED)
        seqs, keys = build_sequences(t)
        pos = pd.Series(np.arange(len(keys)), index=keys)
        mid = m["match_id"].to_numpy()
        home_idx = pos.reindex(pd.MultiIndex.from_arrays([mid, np.ones(len(mid))])).to_numpy()
        away_idx = pos.reindex(pd.MultiIndex.from_arrays([mid, np.zeros(len(mid))])).to_numpy()
        imputer = SimpleImputer(strategy="median").fit(m.loc[tr, cols])
        scaler_x = StandardScaler().fit(imputer.transform(m.loc[tr, cols]))
        static = scaler_x.transform(imputer.transform(m[cols])).astype(np.float32)

        def inputs(mask):
            mask = mask.to_numpy()
            return {"home_seq": seqs[home_idx[mask]], "away_seq": seqs[away_idx[mask]], "static": static[mask]}

        net = lstm(SEQ_LEN, len(SEQ_COLS), len(cols))
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        t1 = time.time()
        net.fit(inputs(tr), y_tr, validation_data=(inputs(va), y_va), epochs=args.epochs, batch_size=512,
                callbacks=[stop], verbose=2)
        scaler = TemperatureScaler().fit(net.predict(inputs(va), batch_size=4096, verbose=0), y_va)
        p = scaler.transform(net.predict(inputs(te), batch_size=4096, verbose=0))
        probs["lstm"] = p
        temps["lstm"] = scaler.t
        results["lstm"] = metrics(y_te, p)
        net.save(MODELS / "lstm.keras")
        joblib.dump({"imputer": imputer, "scaler": scaler_x}, MODELS / "lstm_static_preprocess.joblib")
        print(f"{'lstm':32s} acc {results['lstm']['accuracy']:.4f}  log loss {results['lstm']['log_loss']:.4f}  ({time.time() - t1:.0f}s)")

    for name, p in probs.items():
        results[name]["confusion_matrix"] = confusion_matrix(y_te, p.argmax(1), labels=[0, 1, 2]).tolist()

    split = {
        "train": [str(m.loc[tr, "MatchDate"].min().date()), str(m.loc[tr, "MatchDate"].max().date()), int(tr.sum())],
        "val": [str(m.loc[va, "MatchDate"].min().date()), str(m.loc[va, "MatchDate"].max().date()), int(va.sum())],
        "test": [str(m.loc[te, "MatchDate"].min().date()), str(m.loc[te, "MatchDate"].max().date()), int(te.sum())],
    }
    out = {"labels": LABELS, "split": split, "features": cols, "temperatures": temps, "results": results}
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))
    with open(MODELS / "team_snapshot.pkl", "wb") as f:
        pickle.dump(team_snapshot(t, m.attrs["elo_final"]), f)
    (MODELS / "temperatures.json").write_text(json.dumps(temps, indent=2))
    (MODELS / "features.json").write_text(json.dumps({"features": cols, "features_with_odds": cols_odds}, indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
