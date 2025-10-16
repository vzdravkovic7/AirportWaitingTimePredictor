from src.preprocessing import load_data, add_date_features, clean_data, impute_missing, split_dataset
from src.features import categorize_wait_time, one_hot_encode, normalize_columns
from src.models import train_linear_regression, train_random_forest, train_xgboost, train_mlp_regressor
from src.evaluate import evaluate, save_results, cross_validate_all_metrics, tune_hyperparameters
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.visualizations import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_residuals_boxplot
)


def train_regression(demo=False):
    print("=== REGRESSION PHASE ===")

    df = load_data(demo=demo)
    df = add_date_features(df)
    df = clean_data(df)
    df = categorize_wait_time(df)
    df = one_hot_encode(df, ["AirportCode", "TerminalName", "HourRange", "season"], training=True)
    df = normalize_columns(df, ["TotalPassengerCount", "FlightCount"])

    # df = df.sample(20000, random_state=42)
    y = df["AverageWait"]
    drop_cols = ["AverageWait", "WaitCategory", "FlightDate", "AirportName", "LastUpdated", "SiteId"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = impute_missing(X)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    models = {
        "Linear Regression": train_linear_regression(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train),
        "MLP Regressor": train_mlp_regressor(X_train, y_train),
    }

    all_results = {}

    for name, model in models.items():
        print(f"\n{name} performance:")
        X_val_eval = X_val.values if name == "XGBoost" else X_val
        val_preds = model.predict(X_val_eval)
        val_metrics = evaluate(y_val, val_preds)
        print("Validation metrics:", val_metrics)

        X_eval = X_test.values if name == "XGBoost" else X_test
        preds = model.predict(X_eval)
        test_metrics = evaluate(y_test, preds)
        print("Test metrics:", test_metrics)

        all_results[name] = {"validation": val_metrics, "test": test_metrics}
        cv_results = cross_validate_all_metrics(model, X, y, cv=5)
        all_results[name]["cross_val"] = cv_results

    print("\nStarting hyperparameter tuning...")
    rf_params = {
        "n_estimators": [30, 70, 100],
        "max_depth": [2, 4, 6, 8],
        "min_samples_split": [2, 5, 10]
    }
    xgb_params = {
        "n_estimators": [30, 70, 100],
        "max_depth": [3, 6, 8],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0]
    }

    best_rf, rf_best_params, rf_best_score = tune_hyperparameters(
        RandomForestRegressor(random_state=42), rf_params, X_train, y_train
    )
    print("\nBest Random Forest params:", rf_best_params)
    all_results["Random Forest"]["tuned"] = {"params": rf_best_params, "cv_score": rf_best_score}

    best_xgb, xgb_best_params, xgb_best_score = tune_hyperparameters(
        XGBRegressor(random_state=42, n_jobs=-1), xgb_params, X_train, y_train
    )
    print("\nBest XGBoost params:", xgb_best_params)
    all_results["XGBoost"]["tuned"] = {"params": xgb_best_params, "cv_score": xgb_best_score}

    save_results(all_results, demo=demo)

    # print("\nGenerating regression visualizations...")

    # models_dict = {
    #     "Linear Regression": models["Linear Regression"],
    #     "Random Forest": models["Random Forest"],
    #     "XGBoost": models["XGBoost"],
    #     "MLP Regressor": models["MLP Regressor"],
    # }

    # plot_predictions_vs_actual(models_dict, X_test, y_test)
    # plot_residuals(models_dict, X_test, y_test)
    # plot_residuals_boxplot(models_dict, X_test, y_test)

    print("\nRegression phase completed successfully.")
    return all_results, rf_best_params, xgb_best_params
