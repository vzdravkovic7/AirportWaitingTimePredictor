import os
import numpy as np
import json
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, RandomizedSearchCV


def tune_hyperparameters(model, param_distributions, X, y, cv=3, n_iter=10, scoring="neg_mean_absolute_error"):
    if not isinstance(X, np.ndarray):
        X = X.to_numpy()
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        error_score="raise"
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }


def cross_validate_all_metrics(model, X, y, cv=5):
    if not isinstance(X, np.ndarray):
        X = X.to_numpy()
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()

    scoring = {
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "RMSE": make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False),
        "R2": "r2"
    }

    scores = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise"
    )

    results = {}
    for metric in scoring.keys():
        test_scores = scores[f"test_{metric}"]
        results[metric] = {
            "mean": float(np.mean(test_scores)),
            "std": float(np.std(test_scores)),
            "all_scores": test_scores.tolist()
        }
    return results


def save_results(results, filename="results.json"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}

    existing.update(results)
    existing["timestamp"] = timestamp

    with open(filename, "w") as f:
        json.dump(existing, f, indent=2)

def evaluate_classification(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }
