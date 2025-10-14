import joblib
import os

ARTIFACTS_DIR = "artifacts"
DEFAULT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
CLASS_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_class_model.pkl")

def save_model(model, path=DEFAULT_MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path=DEFAULT_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model

def save_class_model(model, path=CLASS_MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Classification model saved to {path}")

def load_class_model(path=CLASS_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No classification model found at {path}")
    model = joblib.load(path)
    print(f"Classification model loaded from {path}")
    return model