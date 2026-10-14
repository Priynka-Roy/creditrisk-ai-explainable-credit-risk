"""Train-only encoding, standardization and limited development CV."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL


def make_model(X, kind="baseline"):
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )
    estimator = (
        HistGradientBoostingClassifier(random_state=42, max_iter=150, early_stopping=False)
        if kind == "boosting"
        else LogisticRegression(
            penalty=None if kind == "baseline" else "l1" if kind == "l1" else "l2",
            solver="lbfgs" if kind == "baseline" else "liblinear",
            max_iter=3000,
            random_state=42,
        )
    )
    return Pipeline([("preprocessing", transform), ("model", estimator)])


def fit_candidates(X, y):
    candidates = {}
    params = {}
    for kind in ["baseline", "l2", "l1", "boosting"]:
        model = make_model(X, kind)
        if kind == "baseline":
            model.fit(X, y)
            params[kind] = {"penalty": None}
        else:
            grid = (
                {"model__C": [0.01, 0.1, 1, 10]}
                if kind != "boosting"
                else {
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_leaf_nodes": [7, 15],
                    "model__l2_regularization": [1, 10],
                }
            )
            cv = GridSearchCV(
                model,
                grid,
                scoring="neg_log_loss",
                cv=StratifiedKFold(3, shuffle=True, random_state=42),
                n_jobs=1,
                refit=True,
            )
            cv.fit(X, y)
            model = cv.best_estimator_
            params[kind] = {"best_params": cv.best_params_, "cv_log_loss": float(-cv.best_score_)}
        candidates[kind] = model
    return candidates, params


def select_model(results):
    best = min(results, key=lambda k: results[k]["log_loss"])
    for key in ["baseline", "l2", "l1", "boosting"]:
        if (
            results[key]["log_loss"] <= results[best]["log_loss"] + 0.005
            and results[key]["brier"] <= results[best]["brier"] + 0.002
            and results[key]["roc_auc"] >= results[best]["roc_auc"] - 0.01
        ):
            return key
    return best
