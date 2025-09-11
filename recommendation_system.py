import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib

# Use TkAgg backend for interactive plotting
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Config
# -----------------------------
AMENITY_PREFIX = "has"

# Immutable attributes that cannot be changed (e.g., location, room type)
IMMUTABLE_COLS = [
    "Amsterdam", "Barcelona", "Paris", "accommodates_1",
    "accommodates_2", "accommodates_3-4", "accommodates_5-6", "accommodates_7+",
    "bathrooms_0.5-1", "bathrooms_1.5", "bathrooms_2-2.5", "bathrooms_3+",
    "beds_0-1", "beds_2", "beds_3-4", "beds_5+",
    "entire_house", "shared_room_in_house", "hotel/hostel_room"
]

# Excluded groups of features we don’t recommend changes for
EXCLUDE_FROM_RECOMMEND = [
    AMENITY_PREFIX + "_view_core", AMENITY_PREFIX + "_parking_core", AMENITY_PREFIX + "_outdoors_core",
    AMENITY_PREFIX + "_accessibility_core", AMENITY_PREFIX + "_attractions_nearby_core"
]

# Columns irrelevant to recommendations
REMOVE_COLS = [
    "id", "total_host_1", "total_host_2", "total_host_3-5", "total_host_6-20", "total_host_21+"
]

# Feature groups representing buckets (categorical one-hot encoded)
BUCKET_PREFIXES = ["price_", "min_nights_", "img_bright_"]

# Success metrics (define how “successful” a listing is)
SUCCESS_METRICS = {
    "rating": {
        "col": "review_scores_rating",
        "drop": ["estimated_occupancy_l365d", "estimated_revenue_l365d"]
    },
    "occupancy": {
        "col": "estimated_occupancy_l365d",
        "drop": ["review_scores_rating", "estimated_revenue_l365d"]
    },
    "revenue": {
        "col": "estimated_revenue_l365d",
        "drop": ["review_scores_rating", "estimated_occupancy_l365d"]
    }
}
SUCCESS_METRIC_COLS = ["review_scores_rating", "estimated_occupancy_l365d", "estimated_revenue_l365d"]


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_features(df, feature_cols, scaler=None):
    """
    Fill missing values and scale feature columns.

    Args:
        df (pd.DataFrame): Dataframe containing feature columns.
        feature_cols (list[str]): Features to scale.
        scaler (StandardScaler, optional): Fitted scaler to reuse.
                                           If None, a new scaler is trained.

    Returns:
        tuple:
            - X (np.ndarray): Scaled feature matrix.
            - scaler (StandardScaler): Fitted or reused scaler.
    """
    df = df.copy()
    df[feature_cols] = df[feature_cols].fillna(0)  # Replace NaNs with zeros
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(df[feature_cols])
    else:
        X = scaler.transform(df[feature_cols])
    return X, scaler


def filter_similar_pool(df, listing):
    """
    Keep only listings with the same immutable attributes as the test listing.

    Args:
        df (pd.DataFrame): Dataset of listings.
        listing (pd.Series): Test listing row.

    Returns:
        pd.DataFrame: Subset of df with matching immutable attributes.
    """
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
    For each feature, compute the % of neighbors that share the same value
    as the test listing.

    Args:
        neighbors (pd.DataFrame): k nearest successful listings.
        test_listing (pd.Series): Listing being evaluated.
        feature_cols (list[str]): Features to compare.

    Returns:
        dict[str, float]: Agreement percentage for each feature.
    """
    agreements = {}
    for f in feature_cols:
        if f in EXCLUDE_FROM_RECOMMEND or f not in neighbors.columns:
            continue
        test_val = test_listing[f]
        same = (neighbors[f] == test_val).sum()
        total = len(neighbors)
        agreements[f] = 100.0 * same / total if total > 0 else 0.0
    return agreements


def recommend_bucket_features(neighbors, test_listing, bucket_prefixes):
    """
    Suggest moving a listing to the most popular bucket (e.g., price range).

    Args:
        neighbors (pd.DataFrame): Successful listings.
        test_listing (pd.Series): Test listing.
        bucket_prefixes (list[str]): Prefixes that define bucket groups.

    Returns:
        dict[str, str]: Recommendations by bucket group.
    """
    bucket_recs = {}
    for prefix in bucket_prefixes:
        bucket_cols = [col for col in neighbors.columns if col.startswith(prefix)]
        if not bucket_cols:
            continue

        # Popularity = average presence among neighbors
        popularity = neighbors[bucket_cols].mean()
        most_popular_bucket = popularity.idxmax()

        # Identify which bucket test listing belongs to
        test_bucket = [col for col in bucket_cols if test_listing.get(col, 0) == 1]
        test_bucket = test_bucket[0] if test_bucket else None

        # Recommend change if not aligned
        if test_bucket != most_popular_bucket:
            bucket_recs[prefix] = (
                f"Test listing is in '{test_bucket}' but most successful listings "
                f"are in '{most_popular_bucket}'. Recommend switching."
            )
    return bucket_recs


def find_recommendations(add_threshold, current_features, filtered_features, remove_threshold, similar_features):
    """
    Suggest adding/removing features based on neighbor similarity.

    Args:
        add_threshold (float): % of neighbors with a feature to recommend adding.
        current_features (pd.Series): Features of the test listing.
        filtered_features (list[str]): Features considered for recommendation.
        remove_threshold (float): % below which to recommend removal.
        similar_features (pd.Series): Avg feature values among neighbors.

    Returns:
        dict[str, str]: Feature → Recommendation text.
    """
    recommendations = {}
    for col in tqdm(filtered_features, desc="Analyzing features"):
        # Skip bucket-type features (handled separately)
        if any(col.startswith(pref) for pref in BUCKET_PREFIXES):
            continue
        # Recommend adding features common in successful listings
        if current_features[col] < 0.5 and similar_features[col] >= add_threshold:
            recommendations[col] = "Consider ADDING (very common in successful listings)"
        # Recommend removing uncommon features
        elif current_features[col] > 0.5 and similar_features[col] <= remove_threshold:
            if not col.startswith(AMENITY_PREFIX):
                recommendations[col] = "Consider REMOVING (rare in successful listings)"
    return recommendations


def find_knn_from_successful(X_success, X_test_listing, filtered_features, successful, test_listing, top_k):
    """
    Perform kNN search to find most similar successful listings.

    Args:
        X_success (np.ndarray): Features of successful listings.
        X_test_listing (np.ndarray): Features of test listing.
        filtered_features (list[str]): Features used in similarity.
        successful (pd.DataFrame): Successful comparable listings.
        test_listing (pd.Series): Test listing.
        top_k (int): Number of neighbors.

    Returns:
        tuple:
            - current_features (pd.Series): Test listing features.
            - neighbors (pd.DataFrame): Top-k similar listings.
            - similar_features (pd.Series): Avg features of neighbors.
    """
    sim_scores = cosine_similarity(X_test_listing, X_success).flatten()
    top_idx = np.argsort(sim_scores)[-top_k:]
    neighbors = successful.iloc[top_idx]
    similar_features = neighbors[filtered_features].mean()
    current_features = test_listing[filtered_features]
    return current_features, neighbors, similar_features


def preprocess_successful_comparable_listings(feature_cols, rating_col, successful, test_listing, train_df):
    """
    Preprocess features for kNN: scaling + removing success metrics.

    Returns:
        tuple: (X_success, X_test_listing, filtered_features, successful_df)
    """
    numeric_features = [
        c for c in feature_cols
        if np.issubdtype(train_df[c].dtype, np.number) and c not in SUCCESS_METRIC_COLS
    ]
    filtered_features = [c for c in numeric_features if c not in EXCLUDE_FROM_RECOMMEND]

    # Fill NaNs
    train_df[numeric_features] = train_df[numeric_features].fillna(0)
    successful = successful.copy()
    successful[numeric_features] = successful[numeric_features].fillna(0)
    test_listing_filled = test_listing.copy()
    test_listing_filled[numeric_features] = test_listing_filled[numeric_features].fillna(0)

    # Scale all numeric features
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_features])
    X_success = scaler.transform(successful[numeric_features])
    X_test_listing, _ = preprocess_features(pd.DataFrame([test_listing_filled]), numeric_features, scaler)

    return X_success, X_test_listing, filtered_features, successful


def filter_successful_comparable_listings(success_metric, metric_threshold, test_listing, train_df):
    """
    Filter to successful listings comparable to test listing.

    Args:
        success_metric (str): One of {"rating", "occupancy", "revenue"}.
        metric_threshold (float): Percentile cutoff for success.
        test_listing (pd.Series): Listing being evaluated.
        train_df (pd.DataFrame): All listings.

    Returns:
        tuple: (successful listings df, filtered test listing)
    """
    metric_conf = SUCCESS_METRICS[success_metric]
    metric_col = metric_conf["col"]

    successful = filter_similar_pool(train_df, test_listing)
    if metric_col in successful.columns:
        cutoff = successful[metric_col].quantile(metric_threshold)
        successful = successful[successful[metric_col] >= cutoff]

    successful = successful.drop(columns=metric_conf["drop"], errors="ignore")
    test_listing = test_listing.drop(metric_conf["drop"], errors="ignore")
    return successful, test_listing


def normalize_metric_cols(df, metric_cols):
    """
    Normalize success metric columns to 0–1 range.
    """
    for metric in metric_cols:
        if metric in df.columns:
            min_val, max_val = df[metric].min(), df[metric].max()
            if max_val > min_val:
                df[metric] = (df[metric] - min_val) / (max_val - min_val)
            else:
                df[metric] = 0.5  # if all values equal


def recommend_features(train_df, test_listing, feature_cols, top_k=15,
                       success_metric="rating", metric_threshold=0.9,
                       add_threshold=0.9, remove_threshold=0.1):
    """
    Full recommendation step: filter, scale, find neighbors, and suggest improvements.

    Returns:
        dict with recommendations, agreements, and bucket suggestions.
    """
    successful, test_listing = filter_successful_comparable_listings(
        success_metric, metric_threshold, test_listing, train_df
    )
    if successful.empty:
        return {"message": "No similar listings found after filtering."}

    metric_col = SUCCESS_METRICS[success_metric]["col"]
    X_success, X_test_listing, filtered_features, successful = preprocess_successful_comparable_listings(
        feature_cols, metric_col, successful, test_listing, train_df
    )
    current_features, neighbors, similar_features = find_knn_from_successful(
        X_success, X_test_listing, filtered_features, successful, test_listing, top_k
    )
    recommendations = find_recommendations(
        add_threshold, current_features, filtered_features, remove_threshold, similar_features
    )
    agreements = compute_feature_agreements(neighbors, test_listing, filtered_features)
    bucket_recs = recommend_bucket_features(neighbors, test_listing, BUCKET_PREFIXES)

    return {"recommendations": recommendations, "agreements": agreements, "buckets": bucket_recs}


# -----------------------------
# Example Pipeline
# -----------------------------
def run_pipeline(data, random_state=42, success_metric="rating",
                 metric_threshold=0.9, top_k=20, add_threshold=0.80,
                 remove_threshold=0.20, test_rating_threshold=0.5):
    """
    Orchestrates the whole recommendation pipeline.
    See docstring above for steps.
    """
    # Load dataset
    df = pd.read_csv(data) if isinstance(data, str) else data.copy()
    normalize_metric_cols(df, SUCCESS_METRIC_COLS)

    # Select usable features
    feature_cols = [
        col for col in df.columns
        if col not in IMMUTABLE_COLS and col not in EXCLUDE_FROM_RECOMMEND and col not in REMOVE_COLS
    ]

    # Pick test listing
    success_metric_col, test_listing = sample_bad_listing(
        random_state, success_metric, df, test_rating_threshold
    )
    if success_metric_col in test_listing.index:
        test_listing = test_listing.drop(success_metric_col)

    # Run recommender
    recs = recommend_features(df, test_listing, feature_cols, top_k,
                              success_metric, metric_threshold, add_threshold, remove_threshold)
    return recs, test_listing


def sample_bad_listing(random_state, success_metric, test_df, test_rating_threshold):
    """
    Pick a “bad” test listing (below threshold), otherwise random.
    """
    success_metric_col = SUCCESS_METRICS[success_metric]["col"]
    if success_metric_col in test_df.columns:
        test_listing = filter_bad_listing(random_state, success_metric_col, test_df, test_rating_threshold)
    else:
        test_listing = test_df.sample(1).iloc[0]
    test_listing = test_listing.drop(columns=SUCCESS_METRICS[success_metric]["drop"])
    return success_metric_col, test_listing


def filter_bad_listing(random_state, success_metric_col, test_df, test_rating_threshold):
    """
    Pick a test listing below a given success threshold, else random.
    """
    bad_listings = test_df[test_df[success_metric_col] <= test_rating_threshold]
    if bad_listings.empty:
        return test_df.sample(1).iloc[0]
    return test_df.sample(1).iloc[0]

def analyze_all_metrics(csv_path, k_range=range(1, 51), remove_threshold=0.30, add_threshold=0.70):
    """
    Evaluate recommendations across multiple success metrics and k values.

    Returns:
        pd.DataFrame with summary stats per (metric, k).
    """
    all_results = []
    for metric in ["rating", "occupancy", "revenue"]:
        for k in tqdm(k_range, desc=f"Testing k for {metric}"):
            recs, _ = run_pipeline(csv_path, top_k=k, remove_threshold=remove_threshold,
                                   add_threshold=add_threshold, success_metric=metric)
            n_add = sum("ADDING" in v for v in recs["recommendations"].values())
            n_remove = sum("REMOVING" in v for v in recs["recommendations"].values())
            n_bucket = len(recs["buckets"])
            avg_agreement = np.mean(list(recs["agreements"].values())) if recs["agreements"] else 0
            all_results.append({
                "metric": metric, "k": k, "add": n_add, "remove": n_remove,
                "bucket": n_bucket, "avg_agreement": avg_agreement
            })
    return pd.DataFrame(all_results)


def plot_all_metrics(df):
    """
    Plot number of recommendations and agreement vs. k for each success metric.
    """
    df_melt = df.melt(id_vars=["metric", "k"], value_vars=["add", "remove", "bucket"],
                      var_name="type", value_name="count")

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_melt, x="k", y="count", hue="metric", style="type",
                 linewidth=2, markers=True)
    plt.ylabel("Number of Recommendations")
    plt.title("Effect of k on Recommendations (all metrics)")
    plt.legend(title="Metric / Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x="k", y="avg_agreement", hue="metric", linewidth=2)
    plt.ylabel("Average Feature Agreement (%)")
    plt.title("Average Agreement vs k (all metrics)")
    plt.show()


if __name__ == '__main__':
    path = r"C:\Users\hodos\Documents\Uni\Uni-Year-3\Semester2\Data\final_norm_database.csv"
    run_pipeline(path, top_k=25, remove_threshold=0.30, add_threshold=0.70, success_metric="occupancy")

