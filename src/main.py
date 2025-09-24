from preprocessing import load_data, add_date_features, clean_data, save_cleaned_data

def main():
    df = load_data()
    df = add_date_features(df)
    df = clean_data(df)
    save_cleaned_data(df)

if __name__ == "__main__":
    main()
