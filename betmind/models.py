import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def logistic_regression(num_cols, cat_cols=("Division",)):
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), list(num_cols)),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200), list(cat_cols)),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.5, max_iter=2000))])


def random_forest(num_cols, seed=0):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), list(num_cols)),
    ])
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=14, min_samples_leaf=50, max_features="sqrt",
        n_jobs=-1, random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def lstm(seq_len, seq_dim, static_dim, units=32):
    """Two LSTMs (same weights) read each team's last matches, then join with the static features."""
    import keras
    from keras import layers

    home_seq = keras.Input((seq_len, seq_dim), name="home_seq")
    away_seq = keras.Input((seq_len, seq_dim), name="away_seq")
    static = keras.Input((static_dim,), name="static")
    encoder = keras.Sequential([layers.Masking(0.0), layers.LSTM(units)], name="team_encoder")
    x = layers.Concatenate()([encoder(home_seq), encoder(away_seq), static])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(3, activation="softmax")(x)
    model = keras.Model([home_seq, away_seq, static], out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


class TemperatureScaler:
    """Rescales predicted probabilities with one temperature fitted on the validation set."""

    def __init__(self):
        self.t = 1.0

    @staticmethod
    def _apply(p, t):
        logits = np.log(np.clip(p, 1e-9, 1.0)) / t
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, p_val, y_val):
        def nll(t):
            q = self._apply(p_val, t)
            return -np.mean(np.log(q[np.arange(len(y_val)), y_val] + 1e-12))
        self.t = float(minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded").x)
        return self

    def transform(self, p):
        return self._apply(p, self.t)
