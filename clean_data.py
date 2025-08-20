import pandas as pd



def remove_rows(df,new_df_name):
    """
    removes rows from a dataframe with empty values in the required_columns
    :param df:
    :return:
    """
    # # Load CSV and treat blanks as NaN
    # df = pd.read_csv("listings_paris.csv", na_values=["", " "])

    required_columns = [
        "host_id",
        "host_listings_count",
        "neighbourhood_cleansed",
        "property_type",
        "accommodates",
        "bathrooms_text",
        "bedrooms",
        "beds",
        "amenities",
        "price"
    ]

    # Count missing values per required column
    missing_counts = df[required_columns].isna().sum()

    print("Rows missing each column:")
    print(missing_counts)

    # Remove rows where any of these are missing
    df_clean = df.dropna(subset=required_columns)
    print(f"\nOriginal rows: {len(df)}, After cleaning: {len(df_clean)}")
    f"{new_df_name} after cleaning: {len(df_clean)}"
    df_clean.to_csv(f".csv", index=False)
    print("Cleaned data saved to listings_clean.csv")

