import json
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd


def plot_results(results_path="src/results.json"):
    with open(results_path, "r") as f:
        results = json.load(f)

    regression_models = {}
    classification_models = {}

    for model_name, data in results.items():
        if model_name == "timestamp":
            continue
        sample_metrics = list(data.get("validation", {}).keys())
        if any(m.lower().startswith("acc") for m in sample_metrics):
            classification_models[model_name] = data
        else:
            regression_models[model_name] = data

    # Regression Metrics
    if regression_models:
        metrics = ["MAE", "RMSE", "R2"]
        for metric in metrics:
            plt.figure(figsize=(7, 4))
            values = [
                regression_models[m]["validation"].get(metric, None)
                for m in regression_models
            ]
            sb.barplot(x=list(regression_models.keys()), y=values)
            plt.title(f"Regression Models - {metric}")
            plt.ylabel(metric)
            plt.xlabel("Model")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    # Classification Metrics
    if classification_models:
        metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
        for metric in metrics:
            plt.figure(figsize=(7, 4))
            values = [
                classification_models[m]["validation"].get(metric, None)
                for m in classification_models
            ]
            sb.barplot(x=list(classification_models.keys()), y=values)
            plt.title(f"Classification Models - {metric}")
            plt.ylabel(metric)
            plt.xlabel("Model")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


def plot_predictions_vs_actual(models_dict, X_test, y_test):
    plt.figure(figsize=(6 * len(models_dict), 5))
    for i, (name, model) in enumerate(models_dict.items()):
        preds = model.predict(X_test if name != "XGBoost" else X_test.values)
        plt.subplot(1, len(models_dict), i + 1)
        plt.scatter(y_test, preds, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(f"{name}\nPredicted vs Actual")
        plt.xlabel("Actual Wait Time")
        plt.ylabel("Predicted Wait Time")
    plt.tight_layout()
    plt.show()


def plot_residuals(models_dict, X_test, y_test):
    plt.figure(figsize=(6 * len(models_dict), 5))
    for i, (name, model) in enumerate(models_dict.items()):
        preds = model.predict(X_test if name != "XGBoost" else X_test.values)
        residuals = y_test - preds
        plt.subplot(1, len(models_dict), i + 1)
        sb.histplot(residuals, bins=30, kde=True)
        plt.title(f"{name}\nResidual Distribution")
        plt.xlabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.show()


def plot_residuals_boxplot(models_dict, X_test, y_test):
    data = []
    for name, model in models_dict.items():
        preds = model.predict(X_test.values if hasattr(X_test, "values") else X_test)
        residuals = y_test - preds
        for r in residuals:
            data.append({"Model": name, "Residual": r})

    data = pd.DataFrame(data)

    plt.figure(figsize=(8, 5))
    sb.boxplot(x="Model", y="Residual", data=data)
    plt.title("Residuals Comparison Across Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
