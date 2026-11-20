from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MATCHES_CSV = RAW_DIR / "football_matches.csv"

LABELS = ["H", "D", "A"]  # home win, draw, away win
LABEL_TO_ID = {k: i for i, k in enumerate(LABELS)}

# Chronological split: train on older seasons, tune on 2021-22, test on 2022-23 onwards.
VAL_START = pd.Timestamp("2021-07-01")
TEST_START = pd.Timestamp("2022-07-01")


def load_matches(path=MATCHES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["MatchDate"] = pd.to_datetime(df["MatchDate"])
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHome", "FTAway", "FTResult"])
    df = df[df["FTResult"].isin(LABELS)]
    df = df.sort_values(["MatchDate", "Division", "HomeTeam"], kind="mergesort").reset_index(drop=True)
    df["match_id"] = np.arange(len(df))
    df["y"] = df["FTResult"].map(LABEL_TO_ID).astype(int)
    return df


def split_mask(df: pd.DataFrame):
    train = df["MatchDate"] < VAL_START
    val = (df["MatchDate"] >= VAL_START) & (df["MatchDate"] < TEST_START)
    test = df["MatchDate"] >= TEST_START
    return train, val, test
