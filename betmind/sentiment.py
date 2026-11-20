"""Aspect-based sentiment of fan tweets: overall mood plus attack, defence and morale."""

import re

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from betmind.data import RAW_DIR

TWEETS_CSV = RAW_DIR / "football_sentiment.csv"

# Tweet dataset club names -> names used in the match dataset.
CLUB_NAMES = {
    "ManchesterUnited": "Man United", "Liverpool": "Liverpool", "Arsenal": "Arsenal",
    "TottenhamHotspur": "Tottenham", "ManchesterCity": "Man City", "Everton": "Everton",
    "LeicesterCity": "Leicester", "Chelsea": "Chelsea", "CrystalPalace": "Crystal Palace",
    "Southampton": "Southampton", "Burnley": "Burnley", "AFC Bournemouth": "Bournemouth",
}

FOOTBALL_LEXICON = {
    "win": 2.5, "won": 2.5, "victory": 3.0, "champions": 3.0, "brilliant": 3.0, "class": 1.5,
    "clinical": 2.0, "dominant": 2.0, "unbeaten": 2.0, "clean sheet": 2.0, "comeback": 2.0,
    "loss": -2.5, "lost": -2.5, "defeat": -3.0, "bottled": -2.5, "shambles": -3.0, "sacked": -2.0,
    "injury": -2.0, "injured": -2.0, "relegation": -2.5, "dropped": -1.5, "howler": -2.5,
}

ASPECTS = {
    "attack": ["goal", "goals", "scored", "score", "striker", "attack", "finishing", "shot", "chances", "assist"],
    "defence": ["defence", "defense", "defender", "keeper", "goalkeeper", "conceded", "clean sheet", "backline", "tackle"],
    "morale": ["fans", "manager", "coach", "spirit", "confidence", "pressure", "mentality", "atmosphere", "dressing room"],
}


def make_analyzer() -> SentimentIntensityAnalyzer:
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(FOOTBALL_LEXICON)
    return sia


def load_tweets(path=TWEETS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["created_at", "partition_1", "text"])
    df["date"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["team"] = df["partition_1"].map(CLUB_NAMES)
    df = df.dropna(subset=["date", "team", "text"])
    df["text"] = df["text"].str.replace(r"https?://\S+|@\w+", " ", regex=True)
    return df.drop_duplicates(subset=["team", "text"])[["date", "team", "text"]].reset_index(drop=True)


def score_tweets(df: pd.DataFrame) -> pd.DataFrame:
    sia = make_analyzer()
    out = df.copy()
    out["sentiment"] = [sia.polarity_scores(t)["compound"] for t in out["text"]]
    low = out["text"].str.lower()
    for aspect, words in ASPECTS.items():
        pattern = r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
        hit = low.str.contains(pattern, regex=True)
        out[f"sent_{aspect}"] = np.where(hit, out["sentiment"], np.nan)
    return out


def team_sentiment_before(scored: pd.DataFrame, team: str, kickoff: pd.Timestamp, days: int = 7) -> dict:
    """Average sentiment of a club's tweets in the `days` days before kickoff."""
    w = scored[(scored["team"] == team) & (scored["date"] < kickoff) & (scored["date"] >= kickoff - pd.Timedelta(days=days))]
    res = {"n_tweets": len(w), "sentiment": w["sentiment"].mean() if len(w) else np.nan}
    for aspect in ASPECTS:
        res[f"sent_{aspect}"] = w[f"sent_{aspect}"].mean() if len(w) else np.nan
    return res
