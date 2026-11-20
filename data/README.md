# Data

Two CSV files go in `data/raw/`. They are not in the repo (about 340 MB together).

| File | Content | Source |
|---|---|---|
| `football_matches.csv` | 230,557 club matches, 2000–2025, 38 leagues: results, pre-match Elo and form, match stats, betting odds | Kaggle (football match data 2000–2025) |
| `football_sentiment.csv` | ~980,000 tweets about 12 Premier League clubs, 8 July – 12 October 2020 | [EPL Teams Twitter Sentiment Dataset](https://www.kaggle.com/datasets/wjia26/epl-teams-twitter-sentiment-dataset) |

Columns used from `football_matches.csv`:

- `Division`, `MatchDate`, `HomeTeam`, `AwayTeam`
- `FTHome`, `FTAway`, `FTResult`: final score and result. Used for the target and for features of *later* matches only.
- `Form3Home`, `Form5Home`, `Form3Away`, `Form5Away`: points in the previous 3 / 5 matches (checked: they match our own pre-match computation exactly).
- `HomeShots`, `HomeTarget`, `HomeCorners` (and away): only used as rolling averages over previous matches. About half the rows have no shot data.
- `OddHome`, `OddDraw`, `OddAway`: only for the bookmaker baseline and the "with odds" variant.

The dataset's own `HomeElo` / `AwayElo` columns are missing for 39% of rows, so `betmind/features.py` computes its own Elo from results instead.
