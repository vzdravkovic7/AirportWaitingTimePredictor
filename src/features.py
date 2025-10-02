import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

ENCODER_PATH = "artifacts/encoder.pkl"

def categorize_wait_time(df, training=True):
    if training and "AverageWait" in df.columns:
        def label_wait(x):
            if x <= 15:
                return "short"
            elif x <= 45:
                return "medium"
            else:
                return "long"
        df["WaitCategory"] = df["AverageWait"].apply(label_wait)
    return df

def get_or_fit_encoder(df=None, columns=None, training=False):
    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(df[columns])
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(encoder, ENCODER_PATH)
        return encoder
    else:
        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError("Encoder not found, train the model first.")
        return joblib.load(ENCODER_PATH)

def one_hot_encode(df, columns, training=False):
    encoder = get_or_fit_encoder(df, columns, training=training)
    encoded = encoder.transform(df[columns])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(columns),
        index=df.index
    )
    df = df.drop(columns, axis=1)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def normalize_columns(df, columns):
    for col in columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df
