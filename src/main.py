from preprocessing import load_data, add_date_features, clean_data
from features import categorize_wait_time, one_hot_encode, normalize_columns

def main():
    df = load_data()
    df = add_date_features(df)
    df = clean_data(df)
    df = categorize_wait_time(df)

    df = one_hot_encode(df, ["AirportCode", "TerminalName"])

    df = normalize_columns(df, ["TotalPassengerCount", "FlightCount"])

    print(df.head())
    df.to_csv("../data/processed_airport_waits.csv", index=False)

if __name__ == "__main__":
    main()
