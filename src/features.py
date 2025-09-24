import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def categorize_wait_time(df):
    def label_wait(x):
        if x <= 15:
            return "short"
        elif x <= 45:
            return "medium"
        else:
            return "long"
    df["WaitCategory"] = df["AverageWait"].apply(label_wait)
    return df

def one_hot_encode(df, columns):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns))
    df = df.drop(columns, axis=1).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def normalize_columns(df, columns):
    for col in columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df
