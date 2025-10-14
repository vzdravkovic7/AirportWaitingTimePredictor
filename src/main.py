from train_regression import train_regression
from train_classification import train_classification
from persistence import save_model, save_class_model
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from preprocessing import (
    load_data,
    clean_data,
    preprocess_serving,
    preprocess_serving_classification,
    ToNumpy,
)
from features import categorize_wait_time
from visualizations import plot_results


def make_pipeline(model, preprocess_func):
    preprocessing = FunctionTransformer(preprocess_func, validate=False)
    steps = [("preprocessing", preprocessing)]

    if isinstance(model, XGBRegressor):
        steps.append(("to_numpy", ToNumpy()))

    steps.append(("model", model))
    return Pipeline(steps)


def main():
    regression_results, rf_params, xgb_params = train_regression()
    train_classification()

    print("\n=== FINAL SERVING PIPELINE BUILD (Regression) ===")
    raw_df = load_data()
    raw_df = clean_data(raw_df)
    y_full = raw_df["AverageWait"]
    X_full = raw_df[
        ["AirportCode", "TerminalName", "FlightDate", "HourRange", "TotalPassengerCount", "FlightCount"]
    ]

    chosen_model_name = "XGBoost"
    if chosen_model_name == "Linear Regression":
        chosen_model = LinearRegression()
    elif chosen_model_name == "Random Forest":
        chosen_model = RandomForestRegressor(**rf_params, random_state=42)
    elif chosen_model_name == "XGBoost":
        chosen_model = XGBRegressor(**xgb_params, random_state=42, n_jobs=-1)
    elif chosen_model_name == "MLP Regressor":
        chosen_model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=150,
            random_state=42,
        )

    regression_pipeline = make_pipeline(chosen_model, preprocess_serving)
    regression_pipeline.fit(X_full, y_full)
    save_model(regression_pipeline)
    print(f"Saved pipeline with {chosen_model_name} model for serving.")

    print("\n=== FINAL SERVING PIPELINE BUILD (Classification) ===")
    raw_df = load_data()
    raw_df = clean_data(raw_df)
    raw_df = categorize_wait_time(raw_df, training=True)

    X_class = raw_df[
        ["AirportCode", "TerminalName", "FlightDate", "HourRange", "TotalPassengerCount", "FlightCount"]
    ]
    y_class = raw_df["WaitCategory"]

    clf_model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=150,
        random_state=42,
    )

    classification_pipeline = make_pipeline(clf_model, preprocess_serving_classification)
    classification_pipeline.fit(X_class, y_class)
    save_class_model(classification_pipeline)
    print("Saved pipeline with MLP Classifier model for serving.")

    plot_results()


if __name__ == "__main__":
    main()
