import pandas as pd
import glob
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from features import one_hot_encode, normalize_columns

def load_data(path="../data/*.csv"):
    files = glob.glob(path)
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def add_date_features(df):
    if "FlightDate" in df.columns:
        df['FlightDate'] = pd.to_datetime(df['FlightDate'])
        df['day_of_week'] = df['FlightDate'].dt.dayofweek   # 0=Monday
        df['month'] = df['FlightDate'].dt.month
        df['year'] = df['FlightDate'].dt.year

        def get_season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Fall"

        df['season'] = df['month'].apply(get_season)

        df = df.drop(columns=["FlightDate"])
    return df

def clean_data(df, training=True):
    if training:
        if "AverageWait" in df.columns:
            df = df[df['AverageWait'] > 0]
        if "TotalPassengerCount" in df.columns:
            df = df[df['TotalPassengerCount'] > 0]
    df = df.drop_duplicates()
    return df

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=["int64", "float64"]).columns
    
    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy="mean")
        df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])
    
    return df_copy

def preprocess_serving(df):
    df = add_date_features(df)
    df = impute_missing(df)
    df = one_hot_encode(df, ["AirportCode", "TerminalName", "HourRange", "season"], training=False)
    df = normalize_columns(df, ["TotalPassengerCount", "FlightCount"])
    return df

class ToNumpy(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.values if hasattr(X, "values") else X
