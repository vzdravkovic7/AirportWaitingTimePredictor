from train_regression import train_regression
from train_classification import train_classification
from persistence import save_model
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from preprocessing import load_data, clean_data, preprocess_serving, ToNumpy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.neural_network import MLPRegressor


def make_pipeline(model):
    preprocessing = FunctionTransformer(preprocess_serving, validate=False)
    steps = [("preprocessing", preprocessing)]
    if isinstance(model, XGBRegressor):
        steps.append(("to_numpy", ToNumpy()))
    steps.append(("model", model))
    return Pipeline(steps)


def main():
    regression_results, rf_params, xgb_params = train_regression()
    train_classification()

    print("\n=== FINAL SERVING PIPELINE BUILD ===")
    raw_df = load_data()
    raw_df = clean_data(raw_df)
    y_full = raw_df["AverageWait"]
    X_full = raw_df[["AirportCode", "TerminalName", "FlightDate", "HourRange", "TotalPassengerCount", "FlightCount"]]

    chosen_model_name = "MLP Regressor"
    if chosen_model_name == "Linear Regression":
        chosen_model = LinearRegression()
    elif chosen_model_name == "Random Forest":
        chosen_model = RandomForestRegressor(**rf_params, random_state=42)
    elif chosen_model_name == "XGBoost":
        chosen_model = XGBRegressor(**xgb_params, random_state=42, n_jobs=-1)
    elif chosen_model_name == "MLP Regressor":
        chosen_model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )

    pipeline = make_pipeline(chosen_model)
    pipeline.fit(X_full, y_full)
    save_model(pipeline)
    print(f"\nSaved pipeline with {chosen_model_name} model for serving.")


if __name__ == "__main__":
    main()
