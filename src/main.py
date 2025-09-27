from preprocessing import load_data, add_date_features, clean_data, impute_missing
from features import categorize_wait_time, one_hot_encode, normalize_columns
from models import train_linear_regression, train_random_forest, train_xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import json
from datetime import datetime

def save_results(results, filename="results.json"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["timestamp"] = timestamp
    with open(filename, "a") as f:
        f.write(json.dumps(results) + "\n")


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }


def main():
    # Load + preprocess + features
    df = load_data()
    df = add_date_features(df)
    df = clean_data(df)
    df = categorize_wait_time(df)
    df = one_hot_encode(df, ["AirportCode", "TerminalName", "HourRange", "season"])
    df = normalize_columns(df, ["TotalPassengerCount", "FlightCount"])

    # use subset for faster testing
    # df = df.sample(20000, random_state=42)

    # Split features and target
    y = df["AverageWait"]
    drop_cols = ["AverageWait", "WaitCategory", "FlightDate", "AirportName", "LastUpdated", "SiteId"]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    X = impute_missing(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    models = {
        "Linear Regression": train_linear_regression(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train)
    }

    # Evaluate
    all_results = {}
    for name, model in models.items():
        X_eval = X_test.values if name == "XGBoost" else X_test
        preds = model.predict(X_eval)
        metrics = evaluate(y_test, preds)
        print(f"\n{name} performance:")
        print(metrics)
        all_results[name] = metrics

    save_results(all_results)


if __name__ == "__main__":
    main()
