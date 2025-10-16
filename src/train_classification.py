from src.preprocessing import load_data, add_date_features, clean_data, impute_missing, split_dataset
from src.features import categorize_wait_time, one_hot_encode, normalize_columns
from src.models import train_logistic_regression_classifier, train_rf_classifier, train_mlp_classifier
from src.evaluate import evaluate_classification, save_results


def train_classification(demo=False):
    print("\n=== CLASSIFICATION PHASE ===")

    df = load_data(demo=demo)
    df = add_date_features(df)
    df = clean_data(df)
    df = categorize_wait_time(df)
    df = one_hot_encode(df, ["AirportCode", "TerminalName", "HourRange", "season"], training=True)
    df = normalize_columns(df, ["TotalPassengerCount", "FlightCount"])

    # df = df.sample(20000, random_state=42)
    y = df["WaitCategory"]
    drop_cols = ["AverageWait", "WaitCategory", "FlightDate", "AirportName", "LastUpdated", "SiteId"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = impute_missing(X)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    print(f"Classification split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    models = {
        "Logistic Regression (Classifier)": train_logistic_regression_classifier(X_train, y_train),
        "Random Forest (Classifier)": train_rf_classifier(X_train, y_train),
        "MLP Classifier": train_mlp_classifier(X_train, y_train),
    }

    all_results = {}

    for name, model in models.items():
        print(f"\n{name} performance:")
        val_preds = model.predict(X_val)
        val_metrics = evaluate_classification(y_val, val_preds)
        print("Validation metrics:", val_metrics)
        preds = model.predict(X_test)
        test_metrics = evaluate_classification(y_test, preds)
        print("Test metrics:", test_metrics)
        all_results[name] = {"validation": val_metrics, "test": test_metrics}

    save_results(all_results, demo=demo)

    print("\nClassification phase completed successfully.")
