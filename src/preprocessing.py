import pandas as pd
import glob

def load_data(path="../data/*.csv"):
    files = glob.glob(path)
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def add_date_features(df):
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
    return df

def clean_data(df):
    df = df[df['AverageWait'] > 0]
    df = df[df['TotalPassengerCount'] > 0]
    df = df.drop_duplicates()
    return df

def save_cleaned_data(df, output_path="../data/cleaned_airport_waits.csv"):
    df.to_csv(output_path, index=False)
    print(f"Očišćen dataset snimljen u {output_path}")
