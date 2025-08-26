import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
IMMUTABLE_COLS = [
    "bedrooms_group", "city", "area", "room_type", "accommodates_group", "beds_group"
]  # cannot change

EXCLUDE_FROM_RECOMMEND = [
    "has_view_core", "has_parking_core", "has_outdoors_core",
    "has_accessibility_core", "has_attractions_nearby_core"
]

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_features(df, feature_cols, scaler=None):
    df = df.copy()
    # Fill NaNs for all feature columns
    df[feature_cols] = df[feature_cols].fillna(0)
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(df[feature_cols])
    else:
        X = scaler.transform(df[feature_cols])
    return X, scaler



def filter_similar_pool(df, listing):
    """Keep only listings with same immutable attributes as the test listing"""
    cond = pd.Series(True, index=df.index)
    for col in IMMUTABLE_COLS:
        if col in df.columns:
            cond &= (df[col] == listing[col])
    return df[cond].copy()

# -----------------------------
# Recommendation System
# -----------------------------
def recommend_features(train_df, test_listing, feature_cols, top_k=5):
    # Step 1: Filter comparable listings in train
    successful = filter_similar_pool(train_df, test_listing)
    if successful.empty:
        return {"message": "No similar listings found."}

    # Step 2: Feature selection (drop excluded cols)
    filtered_features = [c for c in feature_cols if c not in EXCLUDE_FROM_RECOMMEND]

    # Step 3: Keep only numeric columns
    filtered_features = [c for c in filtered_features if np.issubdtype(train_df[c].dtype, np.number)]

    # Step 4: Scale features using train only
    X_train, scaler = preprocess_features(train_df, filtered_features)
    # After filtering
    successful = successful.copy()
    successful[filtered_features] = successful[filtered_features].fillna(0)

    X_success = scaler.transform(successful[filtered_features])

    X_test_listing, _ = preprocess_features(
        pd.DataFrame([test_listing]), filtered_features, scaler
    )

    # Diagnostic info before similarity
    print("=== Diagnostics ===")
    print("Filtered features:", filtered_features)
    print("X_test_listing shape:", X_test_listing.shape)
    print("X_success shape:", X_success.shape)
    print("NaNs in X_test_listing:", np.isnan(X_test_listing).sum())
    print("NaNs in X_success:", np.isnan(X_success).sum())
    print("===================")

    # Step 5: Compute similarity with try/except
    try:
        sim_scores = cosine_similarity(X_test_listing, X_success).flatten()
    except ValueError as e:
        print("ERROR during cosine_similarity:", e)
        # Show where NaNs are
        print("Columns with NaNs in X_test_listing:")
        for i, col in enumerate(filtered_features):
            if np.isnan(X_test_listing[0, i]):
                print(f" - {col}")
        print("Columns with NaNs in X_success:")
        nan_counts = np.isnan(X_success).sum(axis=0)
        for i, count in enumerate(nan_counts):
            if count > 0:
                print(f" - {filtered_features[i]}: {count} NaNs")
        # Stop execution gracefully
        return {"error": "NaNs detected during similarity computation"}

    top_idx = np.argsort(sim_scores)[-top_k:]
    similar_features = successful.iloc[top_idx][filtered_features].mean()
    current_features = test_listing[filtered_features]

    # Step 6: Recommendations
    recommendations = {}
    for col in tqdm(filtered_features, desc="Analyzing features"):
        if similar_features[col] > 0.5 and current_features[col] < 0.5:
            recommendations[col] = round(similar_features[col], 2)

    return recommendations


# -----------------------------
# Example Pipeline
# -----------------------------
def run_pipeline(csv_path, test_size=0.2, random_state=42):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Define feature columns (all except immutable + excluded + ID)
    feature_cols = [
        col for col in df.columns
        if col not in IMMUTABLE_COLS
           and col not in EXCLUDE_FROM_RECOMMEND
           and col != "id"  # drop the ID column
    ]

    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Pick a random listing from test
    test_listing = test_df.sample(1, random_state=random_state).iloc[0]

    # Generate recommendations
    recs = recommend_features(train_df, test_listing, feature_cols, top_k=5)

    print("\nPicked test listing ID:", test_listing.get("id", "unknown"))
    print("listing", test_listing)
    print("Recommendations:", recs)

    return recs

if __name__ == '__main__':
    path = r"C:\Users\hodos\Documents\Uni\Uni-Year-3\Semester2\Data\freq_item_db.csv"
    run_pipeline(path)
