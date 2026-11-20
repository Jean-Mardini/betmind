"""Pre-match features. Everything here only uses matches played before the one being predicted."""

import numpy as np
import pandas as pd

ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 60.0

ROLL_WINDOWS = (5, 10)
SEQ_LEN = 10
SEQ_COLS = ["gf", "ga", "points", "is_home", "elo_gap", "shots_for", "shots_against", "has_shots"]


def add_elo(df: pd.DataFrame):
    """Elo computed from results, in date order. Stores the rating each team had before kickoff.
    Also returns the ratings after the last match, used by the app."""
    ratings: dict[str, float] = {}
    home_pre = np.empty(len(df))
    away_pre = np.empty(len(df))
    for i, (h, a, gh, ga) in enumerate(zip(df["HomeTeam"], df["AwayTeam"], df["FTHome"], df["FTAway"])):
        rh = ratings.get(h, ELO_START)
        ra = ratings.get(a, ELO_START)
        home_pre[i], away_pre[i] = rh, ra
        expected = 1.0 / (1.0 + 10 ** ((ra - rh - ELO_HOME_ADV) / 400.0))
        score = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
        margin = np.log1p(abs(gh - ga)) + 1.0  # bigger wins move ratings a bit more
        delta = ELO_K * margin * (score - expected)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    df = df.copy()
    df["elo_home"] = home_pre
    df["elo_away"] = away_pre
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df, ratings


def to_team_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, from that team's point of view."""
    common = ["match_id", "MatchDate", "Division"]
    home = pd.DataFrame({
        **{c: df[c] for c in common},
        "team": df["HomeTeam"], "opp": df["AwayTeam"], "is_home": 1.0,
        "gf": df["FTHome"], "ga": df["FTAway"],
        "shots_for": df["HomeShots"], "shots_against": df["AwayShots"],
        "target_for": df["HomeTarget"], "target_against": df["AwayTarget"],
        "corners_for": df["HomeCorners"], "corners_against": df["AwayCorners"],
        "elo": df["elo_home"], "elo_opp": df["elo_away"],
    })
    away = pd.DataFrame({
        **{c: df[c] for c in common},
        "team": df["AwayTeam"], "opp": df["HomeTeam"], "is_home": 0.0,
        "gf": df["FTAway"], "ga": df["FTHome"],
        "shots_for": df["AwayShots"], "shots_against": df["HomeShots"],
        "target_for": df["AwayTarget"], "target_against": df["HomeTarget"],
        "corners_for": df["AwayCorners"], "corners_against": df["HomeCorners"],
        "elo": df["elo_away"], "elo_opp": df["elo_home"],
    })
    t = pd.concat([home, away], ignore_index=True)
    t["points"] = np.select([t["gf"] > t["ga"], t["gf"] == t["ga"]], [3.0, 1.0], 0.0)
    t["elo_gap"] = (t["elo"] - t["elo_opp"]) / 100.0
    t["has_shots"] = t["shots_for"].notna().astype(float)
    return t.sort_values(["team", "MatchDate", "match_id"], kind="mergesort").reset_index(drop=True)


def _past_rolling(g, col: str, window: int) -> pd.Series:
    return g[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())


def team_form_features(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby("team", sort=False)
    out = t[["match_id", "team", "is_home"]].copy()
    for w in ROLL_WINDOWS:
        for col in ["gf", "ga", "points"]:
            out[f"{col}_l{w}"] = _past_rolling(g, col, w)
    for col in ["shots_for", "shots_against", "target_for", "target_against", "corners_for", "corners_against"]:
        out[f"{col}_l5"] = _past_rolling(g, col, 5)
    out["n_prev"] = g.cumcount()
    out["rest_days"] = g["MatchDate"].diff().dt.days.clip(upper=30)
    # Same venue form: home team's last 5 home games, away team's last 5 away games.
    gv = t.groupby(["team", "is_home"], sort=False)
    out["venue_points_l5"] = gv["points"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    # Head to head: this team's average points in its last 5 meetings with the same opponent.
    gh = t.groupby(["team", "opp"], sort=False)
    out["h2h_points_l5"] = gh["points"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    return out


def build_features(df: pd.DataFrame):
    """Returns (match-level feature frame, team rows used for sequences)."""
    df, ratings = add_elo(df)
    t = to_team_rows(df)
    f = team_form_features(t)
    feat_cols = [c for c in f.columns if c not in ("match_id", "team", "is_home")]
    home = f[f["is_home"] == 1.0].set_index("match_id")[feat_cols].add_prefix("home_")
    away = f[f["is_home"] == 0.0].set_index("match_id")[feat_cols].add_prefix("away_")
    m = df.set_index("match_id").join(home).join(away).reset_index()

    m["form3_home"] = m["Form3Home"]
    m["form5_home"] = m["Form5Home"]
    m["form3_away"] = m["Form3Away"]
    m["form5_away"] = m["Form5Away"]
    for w in ROLL_WINDOWS:
        m[f"points_diff_l{w}"] = m[f"home_points_l{w}"] - m[f"away_points_l{w}"]
        m[f"goal_diff_l{w}"] = (m[f"home_gf_l{w}"] - m[f"home_ga_l{w}"]) - (m[f"away_gf_l{w}"] - m[f"away_ga_l{w}"])

    # Bookmaker implied probabilities (margin removed). Used only for the "+ odds" variant and baseline.
    inv = 1.0 / m[["OddHome", "OddDraw", "OddAway"]]
    inv = inv.div(inv.sum(axis=1), axis=0)
    m["odds_p_home"], m["odds_p_draw"], m["odds_p_away"] = inv["OddHome"], inv["OddDraw"], inv["OddAway"]
    m.attrs["elo_final"] = ratings
    return m, t


def feature_columns(m: pd.DataFrame, with_odds: bool = False) -> list[str]:
    cols = ["elo_home", "elo_away", "elo_diff", "form3_home", "form5_home", "form3_away", "form5_away"]
    cols += [c for c in m.columns if c.startswith(("home_", "away_", "points_diff_", "goal_diff_"))]
    if with_odds:
        cols += ["odds_p_home", "odds_p_draw", "odds_p_away"]
    return cols


def build_sequences(t: pd.DataFrame, seq_len: int = SEQ_LEN):
    """For each team row, the previous `seq_len` matches of that team (zero padded, oldest first)."""
    vals = _seq_values(t)
    seqs = np.zeros((len(t), seq_len, len(SEQ_COLS)), np.float32)
    starts = t.groupby("team", sort=False).cumcount().to_numpy()
    for lag in range(1, seq_len + 1):
        idx = np.nonzero(starts >= lag)[0]
        seqs[idx, seq_len - lag] = vals[idx - lag]
    keys = pd.MultiIndex.from_arrays([t["match_id"], t["is_home"]])
    return seqs, keys


def _seq_values(t: pd.DataFrame) -> np.ndarray:
    x = t[SEQ_COLS].copy()
    x[["shots_for", "shots_against"]] = x[["shots_for", "shots_against"]].fillna(0.0) / 10.0
    x[["gf", "ga"]] = x[["gf", "ga"]].clip(upper=6)
    x["points"] = x["points"] / 3.0
    return x.to_numpy(np.float32)


def team_snapshot(t: pd.DataFrame, ratings: dict) -> dict:
    """State of every team after its last match in the data, for predicting a new fixture."""
    vals = _seq_values(t)
    t = t.assign(_row=np.arange(len(t)))
    snap = {}
    for team, g in t.groupby("team", sort=False):
        s = {"division": g["Division"].iloc[-1], "last_date": g["MatchDate"].iloc[-1], "n_prev": len(g),
             "elo": ratings.get(team, ELO_START)}
        for w in ROLL_WINDOWS:
            for col in ["gf", "ga", "points"]:
                s[f"{col}_l{w}"] = g[col].tail(w).mean()
        for col in ["shots_for", "shots_against", "target_for", "target_against", "corners_for", "corners_against"]:
            s[f"{col}_l5"] = g[col].tail(5).mean()
        s["form3"] = g["points"].tail(3).sum()
        s["form5"] = g["points"].tail(5).sum()
        s["home_points_l5"] = g.loc[g["is_home"] == 1.0, "points"].tail(5).mean()
        s["away_points_l5"] = g.loc[g["is_home"] == 0.0, "points"].tail(5).mean()
        rows = g["_row"].to_numpy()[-SEQ_LEN:]
        seq = np.zeros((SEQ_LEN, len(SEQ_COLS)), np.float32)
        seq[SEQ_LEN - len(rows):] = vals[rows]
        s["seq"] = seq
        snap[team] = s
    h2h = t.groupby(["team", "opp"], sort=False)["points"].apply(lambda p: p.tail(5).mean()).to_dict()
    return {"teams": snap, "h2h": h2h}


def fixture_features(home: str, away: str, snap: dict) -> pd.DataFrame:
    """One-row frame with the same columns as training, for a match between two known teams."""
    h, a = snap["teams"][home], snap["teams"][away]
    row = {"Division": h["division"], "elo_home": h["elo"] + 0.0, "elo_away": a["elo"] + 0.0}
    row["elo_diff"] = row["elo_home"] - row["elo_away"]
    row.update({"form3_home": h["form3"], "form5_home": h["form5"], "form3_away": a["form3"], "form5_away": a["form5"]})
    for side, s, venue in (("home", h, "home_points_l5"), ("away", a, "away_points_l5")):
        for w in ROLL_WINDOWS:
            for col in ["gf", "ga", "points"]:
                row[f"{side}_{col}_l{w}"] = s[f"{col}_l{w}"]
        for col in ["shots_for", "shots_against", "target_for", "target_against", "corners_for", "corners_against"]:
            row[f"{side}_{col}_l5"] = s[f"{col}_l5"]
        row[f"{side}_n_prev"] = s["n_prev"]
        row[f"{side}_rest_days"] = 7.0
        row[f"{side}_venue_points_l5"] = s[venue]
    row["home_h2h_points_l5"] = snap["h2h"].get((home, away), np.nan)
    row["away_h2h_points_l5"] = snap["h2h"].get((away, home), np.nan)
    for w in ROLL_WINDOWS:
        row[f"points_diff_l{w}"] = row[f"home_points_l{w}"] - row[f"away_points_l{w}"]
        row[f"goal_diff_l{w}"] = (row[f"home_gf_l{w}"] - row[f"home_ga_l{w}"]) - (row[f"away_gf_l{w}"] - row[f"away_ga_l{w}"])
    return pd.DataFrame([row])
