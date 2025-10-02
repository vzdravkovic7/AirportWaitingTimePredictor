import argparse
import pandas as pd
from persistence import load_model
from fastapi import FastAPI
import uvicorn

app = FastAPI()
model = load_model()

def cli_predict(input_csv: str):
    df = pd.read_csv(input_csv)
    preds = model.predict(df)
    for i, p in enumerate(preds):
        print(f"Sample {i}: Predicted wait time = {p:.2f} minutes")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return {"waiting_time_minutes": round(float(pred[0]), 2)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Airport Waiting Time Predictor")
    parser.add_argument("--csv", type=str, help="Path to input CSV for batch prediction")
    parser.add_argument("--api", action="store_true", help="Run FastAPI server")
    args = parser.parse_args()

    if args.csv:
        cli_predict(args.csv)
    elif args.api:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Please provide --csv <path> for CLI mode or --api for API mode.")
