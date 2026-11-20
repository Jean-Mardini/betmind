"""Small study: does fan sentiment in the week before a match say anything about the result?

The tweet dataset covers 12 Premier League clubs from 8 July to 12 October 2020, so this
only uses the Premier League matches of those clubs in that window.

    python sentiment_study.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from betmind.data import load_matches
from betmind.features import add_elo
from betmind.sentiment import ASPECTS, CLUB_NAMES, load_tweets, score_tweets, team_sentiment_before

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "processed" / "tweets_scored.pkl"
MIN_TWEETS = 50


def main():
    if CACHE.exists():
        scored = pd.read_pickle(CACHE)
    else:
        print("scoring tweets (a few minutes)...")
        scored = score_tweets(load_tweets())
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        scored.to_pickle(CACHE)
    print(f"{len(scored)} tweets, {scored['date'].min().date()} to {scored['date'].max().date()}")

    matches, _ = add_elo(load_matches())
    start, end = scored["date"].min().normalize() + pd.Timedelta(days=7), scored["date"].max()
    epl = matches[(matches["Division"] == "E0") & matches["MatchDate"].between(start, end)]

    clubs = set(CLUB_NAMES.values())
    rows = []
    for _, mt in epl.iterrows():
        for side, team, opp in (("home", mt["HomeTeam"], mt["AwayTeam"]), ("away", mt["AwayTeam"], mt["HomeTeam"])):
            if team not in clubs:
                continue
            s = team_sentiment_before(scored, team, mt["MatchDate"])
            if s["n_tweets"] < MIN_TWEETS:
                continue
            gf, ga = (mt["FTHome"], mt["FTAway"]) if side == "home" else (mt["FTAway"], mt["FTHome"])
            elo_gap = mt["elo_home"] - mt["elo_away"]
            rows.append({
                "date": mt["MatchDate"].date(), "team": team, "opponent": opp, "home": side == "home",
                "points": 3 if gf > ga else 1 if gf == ga else 0, "goal_diff": gf - ga,
                "elo_gap": elo_gap if side == "home" else -elo_gap, **s,
            })
    df = pd.DataFrame(rows)
    print(f"{len(df)} team-matches ({df['team'].nunique()} clubs)")

    # Remove what Elo already explains, then check if sentiment adds anything.
    A = np.c_[np.ones(len(df)), df["elo_gap"]]
    resid = df["points"] - A @ np.linalg.lstsq(A, df["points"], rcond=None)[0]

    out = {"n_team_matches": len(df), "window": [str(start.date()), str(end.date())], "correlations": {}}
    for col in ["elo_gap", "sentiment"] + [f"sent_{a}" for a in ASPECTS]:
        ok = df[col].notna()
        r, p = spearmanr(df.loc[ok, col], df.loc[ok, "points"])
        rr, pr = spearmanr(df.loc[ok, col], resid[ok]) if col != "elo_gap" else (np.nan, np.nan)
        out["correlations"][col] = {"n": int(ok.sum()), "rho_points": round(float(r), 3), "p": round(float(p), 3),
                                    "rho_after_elo": None if np.isnan(rr) else round(float(rr), 3),
                                    "p_after_elo": None if np.isnan(pr) else round(float(pr), 3)}
        print(f"{col:16s} n={ok.sum():3d}  rho={r:+.3f} (p={p:.3f})" + ("" if np.isnan(rr) else f"   after Elo: rho={rr:+.3f} (p={pr:.3f})"))

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "sentiment_study.json").write_text(json.dumps(out, indent=2))
    df.to_csv(ROOT / "results" / "sentiment_study_rows.csv", index=False)


if __name__ == "__main__":
    main()
