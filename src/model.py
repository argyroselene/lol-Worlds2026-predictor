"""Game-outcome models: P(blue side wins).

Models are evaluated with a chronological split (train on older games, test on
the most recent ``test_frac``) to avoid look-ahead leakage.

Candidates
    elo_only       Logistic regression on the team Elo difference (baseline).
    team_logistic  Logistic regression on team features.
    full_logistic  Logistic regression on team and player features.
    full_boosting  Gradient-boosted trees on team and player features.
    ensemble       Mean of full_logistic and full_boosting probabilities.

Training samples are weighted by recency (exponential decay with half-life
HALF_LIFE days), and games from the current Worlds are up-weighted by
CURRENT_WORLDS_WEIGHT.

Metrics: accuracy, log loss and Brier score, overall and on international games.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import CURRENT_WORLDS_START, FEATURE_COLS, INTERNATIONAL, TEAM_FEATURES

MIN_GAMES = 5  # minimum prior games per team for a game to be used
HALF_LIFE = 180  # days; None or 0 gives every game equal weight
CURRENT_WORLDS_WEIGHT = 3.0  # sample-weight multiplier for current-Worlds games


def recency_weights(df, half_life=HALF_LIFE, current_worlds_weight=CURRENT_WORLDS_WEIGHT):
    """Sample weights: exponential decay by age, times a bonus for current-Worlds games."""
    dates = df["date"]
    w = np.ones(len(df)) if not half_life else 0.5 ** ((dates.max() - dates).dt.days.values / half_life)
    current = (df["league"].isin({"WLDs"}) & (dates >= CURRENT_WORLDS_START)).values
    return w * np.where(current, current_worlds_weight, 1.0)


def fit_model(model, X, y, w):
    """Fit a model (estimator, pipeline or Ensemble) with sample weights."""
    if isinstance(model, Ensemble):
        for m in model.models:
            fit_model(m, X, y, w)
    elif hasattr(model, "steps"):  # sklearn pipeline: weight goes to the last step
        model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    else:
        model.fit(X, y, sample_weight=w)
    return model


def _logistic(C=0.5):
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0),
                         StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def _boosting():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=400,
                                          l2_regularization=1.0, min_samples_leaf=80,
                                          early_stopping=True, validation_fraction=0.15,
                                          random_state=0)


class Ensemble:
    """Averages the predicted probabilities of several models."""

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


def make_models():
    return {
        "elo_only": (["diff_elo"], _logistic(C=1.0)),
        "team_logistic": (TEAM_FEATURES, _logistic()),
        "full_logistic": (FEATURE_COLS, _logistic()),
        "full_boosting": (FEATURE_COLS, _boosting()),
        "ensemble": (FEATURE_COLS, Ensemble([_logistic(), _boosting()])),
    }


def time_split(feats, test_frac=0.2):
    df = feats[feats["min_games"] >= MIN_GAMES].sort_values("date")
    cut = df["date"].quantile(1 - test_frac)
    return df[df["date"] < cut], df[df["date"] >= cut], cut


def _scores(y, p, intl):
    out = {"accuracy": accuracy_score(y, p > 0.5), "log_loss": log_loss(y, p),
           "brier": brier_score_loss(y, p)}
    if intl.sum() >= 20:
        out["intl_accuracy"] = accuracy_score(y[intl], p[intl] > 0.5)
        out["intl_log_loss"] = log_loss(y[intl], p[intl], labels=[0, 1])
        out["intl_games"] = int(intl.sum())
    return out


def evaluate(feats, test_frac=0.2, half_life=HALF_LIFE):
    """Fit every candidate on the training period and score it on the test period.

    Returns (scores, info, test_predictions, fitted_models, test_frame).
    """
    train, test, cut = time_split(feats, test_frac)
    y = test["blue_win"].values
    intl = test["league"].isin(INTERNATIONAL).values
    rows, preds, fitted = [], {}, {}
    w = recency_weights(train, half_life)
    for name, (cols, model) in make_models().items():
        fit_model(model, train[cols], train["blue_win"], w)
        p = np.clip(model.predict_proba(test[cols])[:, 1], 1e-6, 1 - 1e-6)
        rows.append({"model": name, **_scores(y, p, intl)})
        preds[name], fitted[name] = p, (cols, model)
    base = np.full(len(test), train["blue_win"].mean())
    rows.append({"model": "always_blue", **_scores(y, base, intl)})
    scores = pd.DataFrame(rows)
    info = {"train_games": len(train), "test_games": len(test), "test_from": str(cut.date()),
            "half_life_days": half_life}
    pred_df = pd.DataFrame({"date": test["date"].values, "league": test["league"].values,
                            "blue_win": y, **preds})
    return scores, info, pred_df, fitted, test


def calibration_table(y, p, bins=10):
    """Mean predicted vs. observed win rate per probability bin."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    t = df.groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"),
                                             games=("y", "size"))
    return t.reset_index(drop=True)


def feature_importance(cols, model, test, n_repeats=5):
    """Permutation importance: increase in test log loss when a feature is shuffled."""
    rng = np.random.default_rng(0)
    X = test[cols].copy()
    y = test["blue_win"].values
    base = log_loss(y, np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6))
    rows = []
    for c in cols:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            drops.append(log_loss(y, np.clip(model.predict_proba(Xp)[:, 1], 1e-6, 1 - 1e-6)) - base)
        rows.append({"feature": c, "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def fit_final(feats, name, half_life=HALF_LIFE):
    """Refit the chosen model on all eligible games."""
    df = feats[feats["min_games"] >= MIN_GAMES]
    cols, model = make_models()[name]
    fit_model(model, df[cols], df["blue_win"], recency_weights(df, half_life))
    return cols, model
