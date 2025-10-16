import argparse
import pandas as pd
from src.persistence import load_model, load_class_model
from fastapi import FastAPI, UploadFile, File
import uvicorn
import io

app = FastAPI(title="Airport Waiting Time Predictor API")

model = load_model()
classification_model = load_class_model()


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


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    preds = model.predict(df)
    df["PredictedWaitTime"] = preds
    result = df[["PredictedWaitTime"]].to_dict(orient="records")
    return {"batch_predictions": result}


@app.post("/predict_class")
def predict_class(data: dict):
    df = pd.DataFrame([data])
    pred_class = classification_model.predict(df)[0]
    return {"predicted_category": str(pred_class)}


@app.post("/predict_class_batch")
async def predict_class_batch(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    preds = classification_model.predict(df)
    df["PredictedCategory"] = preds
    result = df[["PredictedCategory"]].to_dict(orient="records")
    return {"batch_class_predictions": result}


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
