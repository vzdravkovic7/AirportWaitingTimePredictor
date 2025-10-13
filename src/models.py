from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier

def train_linear_regression(X_train, y_train):
    print("Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=12):
    print("Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    print("Training XGBoost...")
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    X_in = X_train.values if hasattr(X_train, "values") else X_train
    y_in = y_train.values if hasattr(y_train, "values") else y_train

    model.fit(X_in, y_in, verbose=True)
    return model

def train_logistic_regression_classifier(X_train, y_train):
    print("Training Logistic Regression (Classifier)...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_rf_classifier(X_train, y_train, n_estimators=200, max_depth=15):
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_mlp_regressor(X_train, y_train):
    print("Training MLP Regressor...")
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_mlp_classifier(X_train, y_train):
    print("Training MLP Classifier...")
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model