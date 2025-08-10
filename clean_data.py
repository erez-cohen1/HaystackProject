import pandas as pd
import ast


def load_csv(path):
    """Load CSV file from the given path."""
    return pd.read_csv(path)


def process_minimum_nights(df, x):
    """If < x invalids, drop rows; else replace with 1."""
    col = "minimum_nights"
    invalid_mask = df[col].isna() | (df[col] < 0)
    invalid_count = invalid_mask.sum()
    if invalid_count < x:
        df = df[~invalid_mask]
    else:
        df.loc[invalid_mask, col] = 1
    return df


def process_review_columns(df, x):
    """For review columns: if > x invalids → replace with 0, else drop rows."""
    review_cols = ["number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d"]
    invalid_mask = (df[review_cols].isna()) | (df[review_cols] < 0)
    invalid_count = invalid_mask.sum().max()  # worst-case col

    if invalid_count > x:
        for col in review_cols:
            mask = df[col].isna() | (df[col] < 0)
            df.loc[mask, col] = 0
    else:
        for col in review_cols:
            df = df[~(df[col].isna() | (df[col] < 0))]
    return df


def process_binary_columns(df):
    """Replace invalid binary values with 0 (False)."""
    binary_cols = ["host_is_superhost", "host_has_profile_pic",
                   "host_identity_verified", "instant_bookable", "has_availability"]
    for col in binary_cols:
        df[col] = df[col].map(lambda v: 1 if str(v).strip().lower() in ["1", "true", "t", "yes", "y"] else 0)
    return df


def add_host_has_verifications(df):
    """Add column host_has_verifications based on list length."""

    def count_verifications(val):
        try:
            lst = ast.literal_eval(val) if isinstance(val, str) else []
            return 1 if len(lst) > 0 else 0
        except Exception:
            return 0

    df["host_has_verifications"] = df["host_verifications"].apply(count_verifications)
    return df


def clean_airbnb_csv(path, x):
    df = load_csv(path)
    df = process_minimum_nights(df, x)
    df = process_review_columns(df, x)
    df = process_binary_columns(df)
    df = add_host_has_verifications(df)
    return df


# Example usage:
if __name__ == "__main__":
    cleaned_df = clean_airbnb_csv(r"C:\Users\hodos\Documents\Uni\Uni-Year-3\Semester2\Data\listings.csv", x=10)
    cleaned_df.to_csv(r"C:\Users\hodos\Documents\Uni\Uni-Year-3\Semester2\Data\cleaned_listings.csv", index=False)
