# BetMind

Predicting football match results (home win, draw, away win) using only information available before kickoff, plus a small study of whether fan sentiment on Twitter says anything about the result.

NLP course project, Université Saint-Joseph de Beyrouth (ESIB), Fall 2025, supervised by Dr. Nabil Abdoun. Team: Jean Mardini, Marc Khattar, Mike Chedid.

This repository is a cleaned-up version of the course project. The training pipeline was rewritten so that every feature is known before the match starts, and the models are tested on seasons that come after the ones they are trained on.

## Results

Test set: 33,237 matches from July 2022 to June 2025 in 38 leagues. Models are trained on 2000–2021 and calibrated (temperature scaling) on the 2021–22 season.

| Model | Accuracy | Macro F1 | Log loss | Brier |
|---|:---:|:---:|:---:|:---:|
| Always predict a home win | 43.8% | 0.203 | 1.074 | 0.650 |
| Logistic regression | 49.3% | 0.367 | 1.018 | 0.610 |
| Random forest | 49.0% | 0.352 | 1.019 | 0.610 |
| LSTM over each team's last 10 matches | 49.2% | 0.359 | 1.017 | 0.609 |
| Bookmaker odds (margin removed) | 50.4% | 0.373 | 1.001 | 0.598 |
| Logistic regression + bookmaker odds | 50.5% | 0.382 | 1.001 | 0.599 |

Log loss for the "always home" row uses the class frequencies of the training set as probabilities.

What we take from it:

- The three models are close to each other: about 49%, 5.5 points above always picking the home team and about 1 point below the bookmakers. With the same inputs, the choice of model matters little.
- Adding the bookmaker odds as features brings logistic regression level with the bookmakers but not above them.
- Draws are 26% of matches but almost never the most likely outcome, so the models nearly never predict one. This is the main source of error.
- The LSTM reads each team's last 10 matches as a sequence. It gives the best probabilities among the models without odds (lowest log loss), but no accuracy gain over logistic regression.

Exact numbers are in [`results/metrics.json`](results/metrics.json).

## How it works

**Data.** 230,557 club matches from 2000 to 2025 (results, betting odds, shots and corners when available) and about 980,000 tweets about 12 Premier League clubs from July to October 2020 (664,068 after removing duplicates). See [`data/README.md`](data/README.md).

**Features** (`betmind/features.py`). 43 per match, all computed from earlier matches only:

- Elo ratings computed from results in date order (the dataset's own Elo column is missing for 39% of matches)
- points in the last 3 and 5 matches, goals scored and conceded over the last 5 and 10, home form of the home team and away form of the away team
- head-to-head points over the last 5 meetings, days since the previous match, number of previous matches in the data
- average shots, shots on target and corners over the last 5 matches, where the data has them (missing values get an indicator)

The final score, half-time score and in-match statistics of the match itself are never used as inputs.

**Models** (`betmind/models.py`). Logistic regression with the league as a one-hot feature, a random forest (300 trees), and a Keras LSTM. The LSTM encodes the last 10 matches of each team with a shared LSTM layer and combines them with the 43 features. All probabilities are calibrated with one temperature fitted on the validation season.

**App** (`app.py`). A Gradio interface: choose a home and an away team (1,217 teams, common spellings such as "Man Utd" are recognised) and get the probabilities from each model and their average, based on each team's latest matches in the data.

## Sentiment study

`sentiment_study.py` scores the tweets with VADER plus a small football lexicon and keeps three aspects (attack, defence, morale) using keyword lists. For each Premier League match of the 12 clubs between July and October 2020, it averages the club's tweets from the 7 days before the match and compares that with the result.

On the 76 team-matches available there is no measurable link: Spearman correlation with points earned is −0.02 for overall sentiment and between +0.04 and +0.06 for the three aspects, none significant (all p > 0.5). Elo difference on the same matches gives +0.20 (p = 0.08). The sample is small, so this only rules out a strong effect. Results are in [`results/sentiment_study.json`](results/sentiment_study.json).

## Run it

```bash
git clone https://github.com/Jean-Mardini/betmind.git
cd betmind
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the two CSV files in `data/raw/` (see [`data/README.md`](data/README.md)), then:

```bash
python train.py                    # about 4 minutes on a laptop CPU; writes models/ and results/metrics.json
python sentiment_study.py          # about 5 minutes, most of it scoring the tweets
python app.py                      # opens the app on http://127.0.0.1:7860
```

## Project structure

```text
betmind/
├── betmind/
│   ├── data.py            # loading, labels, train/validation/test split by date
│   ├── features.py        # Elo, rolling form, head-to-head, sequences for the LSTM
│   ├── models.py          # logistic regression, random forest, LSTM, temperature scaling
│   └── sentiment.py       # tweet cleaning, VADER + aspects
├── train.py               # trains and evaluates every model
├── sentiment_study.py     # sentiment vs results, Premier League 2020
├── app.py                 # Gradio app
├── data/README.md         # where to get the data
└── results/               # metrics.json, sentiment_study.json
```
