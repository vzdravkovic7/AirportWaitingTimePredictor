from preprocessing import load_data, add_date_features, clean_data, impute_missing, preprocess_serving, ToNumpy
from features import categorize_wait_time, one_hot_encode, normalize_columns
from models import train_linear_regression, train_random_forest, train_xgboost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from persistence import save_model
from xgboost import XGBRegressor
from evaluate import evaluate, save_results, cross_validate_all_metrics, tune_hyperparameters

def make_pipeline(model):
    preprocessing = FunctionTransformer(preprocess_serving, validate=False)
    steps = [("preprocessing", preprocessing)]
    if isinstance(model, XGBRegressor):
        steps.append(("to_numpy", ToNumpy()))
    steps.append(("model", model))
    return Pipeline(steps)

def main():
    # Load & preprocess
    df = load_data()
    df = add_date_features(df)
    df = clean_data(df)
    df = categorize_wait_time(df)
    df = one_hot_encode(df, ["AirportCode", "TerminalName", "HourRange", "season"], training=True)
    df = normalize_columns(df, ["TotalPassengerCount", "FlightCount"])

    df = df.sample(20000, random_state=42)
    y = df["AverageWait"]
    drop_cols = ["AverageWait", "WaitCategory", "FlightDate", "AirportName", "LastUpdated", "SiteId"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = impute_missing(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train baseline models
    models = {
        "Linear Regression": train_linear_regression(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train),
    }

    # Evaluate models
    all_results = {}
    for name, model in models.items():
        X_eval = X_test.values if name == "XGBoost" else X_test
        preds = model.predict(X_eval)
        metrics = evaluate(y_test, preds)
        print(f"\n{name} performance (baseline):")
        print(metrics)
        all_results[name] = {"baseline": metrics}

        cv_results = cross_validate_all_metrics(model, X, y, cv=5)
        all_results[name]["cross_val"] = cv_results

    # Hyperparameter tuning
    print("\nStarting hyperparameter tuning...")
    rf_params = {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]}
    xgb_params = {"n_estimators": [100, 200, 300], "max_depth": [3, 6, 10], "learning_rate": [0.01, 0.1, 0.2], "subsample": [0.7, 0.8, 1.0]}

    best_rf, rf_best_params, rf_best_score = tune_hyperparameters(RandomForestRegressor(random_state=42), rf_params, X_train, y_train)
    print("\nBest Random Forest params:", rf_best_params)
    all_results["Random Forest"]["tuned"] = {"params": rf_best_params, "cv_score": rf_best_score}

    best_xgb, xgb_best_params, xgb_best_score = tune_hyperparameters(XGBRegressor(random_state=42, n_jobs=-1), xgb_params, X_train, y_train)
    print("\nBest XGBoost params:", xgb_best_params)
    all_results["XGBoost"]["tuned"] = {"params": xgb_best_params, "cv_score": xgb_best_score}

    save_results(all_results)

    # Final pipeline for serving
    raw_df = load_data()
    raw_df = clean_data(raw_df)
    y_full = raw_df["AverageWait"]

    X_full = raw_df[["AirportCode", "TerminalName", "FlightDate", "HourRange", "TotalPassengerCount", "FlightCount"]]

    chosen_model_name = "XGBoost"
    if chosen_model_name == "Linear Regression":
        chosen_model = LinearRegression()
    elif chosen_model_name == "Random Forest":
        chosen_model = RandomForestRegressor(**rf_best_params, random_state=42)
    else:
        chosen_model = XGBRegressor(**xgb_best_params, random_state=42, n_jobs=-1)

    pipeline = make_pipeline(chosen_model)
    pipeline.fit(X_full, y_full)

    save_model(pipeline)
    print(f"\nSaved pipeline with {chosen_model_name} model for serving.")

if __name__ == "__main__":
    main()
