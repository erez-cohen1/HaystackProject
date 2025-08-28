import pandas as pd



def remove_rows(df):
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
    return df.dropna(subset=required_columns)

def convert_bathroom_text(df):
    """
    take from bathroom text num of bathroom and put in the bathrooms
    :param df:
    :return: df with the coloumns: 'id', 'bathrooms', 'bathroom_private' (1 if yes)
    """
    # Make a copy to avoid warnings
    df = df.copy()

    # print_rows_per_val(df['estimated_occupancy_l365d'])
    # plot_dist(df['estimated_occupancy_l365d'])
    # text to num
    string_col = 'bathrooms_text'  # column with strings containing numbers
    target_col = 'bathrooms'  # column that might be empty

    df['extracted_number'] = df[string_col].str.extract(r'(\d+)')  # Note the r prefix

    # Fill empty target_column values
    df[target_col] = df[target_col].fillna(df['extracted_number'])
    return df

def clean_db(df,new_df_name='cleaned_database'):

    orig_len=len(df)
    df = remove_rows(df)
    df=convert_bathroom_text(df)

    print(f"\nOriginal rows: {orig_len}, After cleaning: {len(df)}")
    df.to_csv(f"{new_df_name}.csv", index=False)
    print(f"Cleaned data saved to {new_df_name}.csv")
    return df
