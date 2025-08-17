import pandas as pd

# Load CSV and treat blanks as NaN
df = pd.read_csv("listings.csv", na_values=["", " "])

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
df_clean.to_csv("listings_clean.csv", index=False)
print("Cleaned data saved to listings_clean.csv")

