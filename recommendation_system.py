import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

# -----------------------------
# Config
# -----------------------------
IMMUTABLE_COLS = [
    "bedrooms_group", "city", "area", "room_type", "accommodates_group",
    "beds_group", "Amsterdam", "Barcelona", "Paris"
]  # cannot change

EXCLUDE_FROM_RECOMMEND = [
    "has_view_core", "has_parking_core", "has_outdoors_core",
    "has_accessibility_core", "has_attractions_nearby_core"
]

REMOVE_COLS = [
    "estimated_occupancy_l365d", "estimated_revenue_l365d", "id", "total_host_1", "total_host_2",
    "total_host_3-5", "total_host_6-20", "total_host_21+"
]

BUCKET_PREFIXES = ["beds_", "bathrooms", "price_", "accommodates_", "min_nights_"]


# -----------------------------
# Visualization Functions
# -----------------------------

def plot_feature_bar(test_listing, successful, features, top_k=None):
    """
    Bar plot showing % of successful listings with each feature, overlaying the test listing value
    """
    if top_k is not None:
        successful = successful.iloc[top_k]
    feature_percent = successful[features].mean() * 100
    test_values = test_listing[features]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_percent.index, y=feature_percent.values, color='skyblue')
    plt.scatter(range(len(features)), test_values * 100, color='red', s=100, label="Test Listing")
    plt.xticks(rotation=90)
    plt.ylabel("% of successful listings with feature")
    plt.title("Feature comparison: Successful listings vs Test listing")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_dot_lollipop(test_listing, successful, features, top_k=None):
    """
    Dot plot (lollipop style) showing % of successful listings with feature
    """
    if top_k is not None:
        successful = successful.iloc[top_k]
    feature_percent = successful[features].mean() * 100
    test_values = test_listing[features]

    plt.figure(figsize=(12, 6))
    plt.hlines(y=feature_percent.index, xmin=0, xmax=feature_percent.values, color='skyblue', lw=4)
    plt.scatter(feature_percent.values, feature_percent.index, color='skyblue', s=200)
    for i, f in enumerate(features):
        if test_values[f] > 0.5:
            plt.scatter(100, f, color='red', marker='X', s=100)
    plt.xlabel("% of successful listings with feature")
    plt.title("Lollipop plot: Test listing vs successful listings")
    plt.tight_layout()
    plt.show()


def plot_recommendation_map(test_listing, successful, features, add_threshold=0.9, remove_threshold=0.1, top_k=None):
    """
    Color-coded map showing which features should be added, removed, or kept
    """
    if top_k is not None:
        successful = successful.iloc[top_k]
    feature_percent = successful[features].mean()
    test_values = test_listing[features]

    colors = []
    for f in features:
        if test_values[f] < 0.5 and feature_percent[f] >= add_threshold:
            colors.append("blue")  # add
        elif test_values[f] > 0.5 and feature_percent[f] <= remove_threshold:
            colors.append("red")  # remove
        else:
            colors.append("gray")  # keep

    plt.figure(figsize=(12, 1))
    plt.bar(range(len(features)), [1] * len(features), color=colors)
    plt.xticks(range(len(features)), features, rotation=90)
    plt.yticks([])
    plt.title("Recommendation Map (Blue=Add, Red=Remove, Gray=Keep)")
    plt.tight_layout()
    plt.show()


def plot_parallel_coordinates(test_listing, successful, features, top_k=10):
    """
    Parallel coordinates plot: each line = a listing, test listing highlighted
    """
    top_successful = successful.iloc[:top_k].copy()
    top_successful["type"] = "successful"
    test_df = pd.DataFrame([test_listing[features]])
    test_df["type"] = "test"

    combined = pd.concat([top_successful[features + ["type"]], test_df], ignore_index=True)
    plt.figure(figsize=(14, 6))
    parallel_coordinates(combined, "type", color=["skyblue", "red"], alpha=0.7)
    plt.xticks(rotation=90)
    plt.ylabel("Feature value (0/1)")
    plt.title("Parallel Coordinates: Test vs Successful Listings")
    plt.tight_layout()
    plt.show()


def visualize(test_listing, successful, features, top_k=10,
              add_threshold=0.9, remove_threshold=0.1):
    plot_feature_bar(test_listing, successful, features, top_k=top_k)
    plot_dot_lollipop(test_listing, successful, features, top_k=top_k)
    plot_recommendation_map(test_listing, successful, features, top_k=top_k,
                            add_threshold=add_threshold, remove_threshold=remove_threshold)
    plot_parallel_coordinates(test_listing, successful, features, top_k=top_k)


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
def compute_feature_agreements(neighbors, test_listing, feature_cols):
    """
    For each feature, compute the % of neighbors that agree with the test listing.
    """
    agreements = {}
    for f in feature_cols:
        if f in EXCLUDE_FROM_RECOMMEND:  # ignore excluded features
            continue

        if f not in neighbors.columns:
            continue

        test_val = test_listing[f]
        same = (neighbors[f] == test_val).sum()
        total = len(neighbors)
        agreements[f] = 100.0 * same / total if total > 0 else 0.0
    return agreements


def recommend_bucket_features(neighbors, test_listing, bucket_prefixes):
    """
    For each bucket group (by prefix), recommend the most popular bucket
    among neighbors if the test listing is not in it.
    """
    bucket_recs = {}

    for prefix in bucket_prefixes:
        # Get all columns that belong to this bucket group
        bucket_cols = [col for col in neighbors.columns if col.startswith(prefix)]
        if not bucket_cols:
            continue

        # Compute the popularity of each bucket among neighbors (mean ~ percentage of listings with it)
        popularity = neighbors[bucket_cols].mean()

        # Find the most popular bucket column
        most_popular_bucket = popularity.idxmax()

        # Find which bucket the test listing currently belongs to
        test_bucket = [col for col in bucket_cols if test_listing.get(col, 0) == 1]
        test_bucket = test_bucket[0] if test_bucket else None

        # Recommend change if test listing bucket != most popular
        if test_bucket != most_popular_bucket:
            bucket_recs[prefix] = (
                f"Test listing is in '{test_bucket}' but most successful listings "
                f"are in '{most_popular_bucket}'. Recommend switching."
            )
        else:
            print("test listing is in the most popular bucket", most_popular_bucket)

    return bucket_recs


def recommend_features(train_df, test_listing, feature_cols, top_k=15, min_rating=None,
                       rating_col="review_scores_rating", add_threshold=0.9, remove_threshold=0.1):
    successful, test_listing = filter_successful_comparable_listings(min_rating, rating_col, test_listing, train_df)

    if successful.empty:
        return {"message": "No similar listings found after filtering."}

    X_success, X_test_listing, filtered_features, successful = preprocess_successful_comparable_listings(feature_cols,
                                                                                                         rating_col,
                                                                                                         successful,
                                                                                                         test_listing,
                                                                                                         train_df)
    # Step 5: Compute similarity
    current_features, neighbors, similar_features = find_knn_from_successful(X_success, X_test_listing,
                                                                             filtered_features, successful,
                                                                             test_listing, top_k)

    # Step 6: Recommendations
    recommendations = find_recommendations(add_threshold, current_features, filtered_features, remove_threshold,
                                           similar_features)

    # Step 7: Feature Agreement Analysis
    agreements = compute_feature_agreements(neighbors, test_listing, filtered_features)
    bucket_recs = recommend_bucket_features(neighbors, test_listing, BUCKET_PREFIXES)
    print("\n=== Feature Agreement with Neighbors ===")
    for f, pct in agreements.items():
        print(f"{f}: {pct:.1f}%")
    print("\n=== Bucket Recommendations ===")
    for prefix, rec in bucket_recs.items():
        print(f"{prefix}: {rec}")
    for feature, msg in recommendations.items():
        print(f"{feature}: {msg}")
    return {
        "recommendations": recommendations,
        "agreements": agreements,
        "buckets": bucket_recs
    }


def find_recommendations(add_threshold, current_features, filtered_features, remove_threshold, similar_features):
    recommendations = {}
    for col in tqdm(filtered_features, desc="Analyzing features"):
        if current_features[col] < 0.5 and similar_features[col] >= add_threshold:
            recommendations[col] = "Consider ADDING (very common in successful listings)"
        elif current_features[col] > 0.5 and similar_features[col] <= remove_threshold:
            recommendations[col] = "Consider REMOVING (rare in successful listings)"
    return recommendations


def find_knn_from_successful(X_success, X_test_listing, filtered_features, successful, test_listing, top_k):
    sim_scores = cosine_similarity(X_test_listing, X_success).flatten()
    top_idx = np.argsort(sim_scores)[-top_k:]
    neighbors = successful.iloc[top_idx]
    similar_features = neighbors[filtered_features].mean()
    current_features = test_listing[filtered_features]
    return current_features, neighbors, similar_features


def preprocess_successful_comparable_listings(feature_cols, rating_col, successful, test_listing, train_df):
    # Step 2: Keep all numeric columns for similarity (including excluded)
    numeric_features = [
        c for c in feature_cols
        if np.issubdtype(train_df[c].dtype, np.number) and c != rating_col
    ]
    # Step 3: Features for recommendation (exclude excluded features)
    filtered_features = [c for c in numeric_features if c not in EXCLUDE_FROM_RECOMMEND]
    # Step 4: Scale features using train only (all numeric)
    X_train, scaler = preprocess_features(train_df, numeric_features)
    successful = successful.copy()
    numeric_features = [col for col in numeric_features if col in successful.columns]
    successful[numeric_features] = successful[numeric_features].fillna(0)
    X_success = scaler.transform(successful[numeric_features])
    X_test_listing, _ = preprocess_features(pd.DataFrame([test_listing]), numeric_features, scaler)
    return X_success, X_test_listing, filtered_features, successful


def filter_successful_comparable_listings(min_rating, rating_col, test_listing, train_df):
    # Step 1: Filter comparable listings in train
    successful = filter_similar_pool(train_df, test_listing)
    # Step 1.5: Apply rating filter if provided
    if min_rating is not None and rating_col in successful.columns:
        successful = successful[successful[rating_col] >= min_rating]
    # Remove rating column after filtering
    if rating_col in successful.columns:
        successful = successful.drop(columns=[rating_col])
    if rating_col in test_listing.index:
        test_listing = test_listing.drop(rating_col)
    return successful, test_listing


# -----------------------------
# Example Pipeline
# -----------------------------
def run_pipeline(csv_path, test_size=0.2, random_state=40, min_rating=4.9, top_k=20,
                 add_threshold=0.80, remove_threshold=0.20, test_rating_threshold=3.0):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Define feature columns (all except immutable + excluded + ID)
    feature_cols = [
        col for col in df.columns
        if col not in IMMUTABLE_COLS
           and col not in EXCLUDE_FROM_RECOMMEND
           and col not in REMOVE_COLS
    ]

    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Keep only “unsuccessful” listings in test set
    if "review_scores_rating" in test_df.columns:
        low_rating_test_df = test_df[test_df["review_scores_rating"] <= test_rating_threshold]
        if low_rating_test_df.empty:
            print(f"No test listings below rating threshold {test_rating_threshold}. Using random test listing.")
            test_listing = test_df.sample(1, random_state=random_state).iloc[0]
        else:
            test_listing = low_rating_test_df.sample(1, random_state=random_state).iloc[0]
    else:
        test_listing = test_df.sample(1, random_state=random_state).iloc[0]

    # Remove rating column from the test listing
    if "review_scores_rating" in test_listing.index:
        print("test listing rating ", test_listing["review_scores_rating"])
        test_listing = test_listing.drop("review_scores_rating")

    # Generate recommendations with rating filter
    recs = recommend_features(train_df, test_listing, feature_cols, top_k=top_k, min_rating=min_rating,
                              add_threshold=add_threshold, remove_threshold=remove_threshold)

    print("\nPicked test listing ID:", test_listing.get("id", "unknown"))

    return recs



if __name__ == '__main__':
    path = r"C:\Users\hodos\Documents\Uni\Uni-Year-3\Semester2\Data\final_norm_database.csv"
    run_pipeline(path, min_rating=4.98, top_k=25, remove_threshold= 30, add_threshold=70)  # Example: only keep listings with rating >= 80
