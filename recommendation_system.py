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
AMENITY_PREFIX = "has"

IMMUTABLE_COLS = [
    "bedrooms_group", "city", "area", "room_type", "accommodates_group",
    "beds_group", "Amsterdam", "Barcelona", "Paris", "accommodates_1",
    "accommodates_2", "accommodates_3-4", "accommodates_5-6", "accommodates_7+",
    "bathrooms_0.5-1", "bathrooms_1.5", "bathrooms_2-2.5", "bathrooms_3+", "beds_0-1", "beds_2", "beds_3-4", "beds_5+",
    "entire_house", "shared_room_in_house", "hotel/hostel_room"
]  # cannot change

EXCLUDE_FROM_RECOMMEND = [
    AMENITY_PREFIX + "_view_core", AMENITY_PREFIX + "_parking_core", AMENITY_PREFIX + "_outdoors_core",
    AMENITY_PREFIX + "_accessibility_core", AMENITY_PREFIX + "_attractions_nearby_core"
]

REMOVE_COLS = [
    "id", "total_host_1", "total_host_2", "total_host_3-5", "total_host_6-20", "total_host_21+"
]
BUCKET_PREFIXES = ["price_", "min_nights_"]

# -----------------------------
# Config for success definitions
# -----------------------------
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


def recommend_features(train_df, test_listing, feature_cols, top_k=15,
                       success_metric="rating", metric_threshold=0.9,
                       add_threshold=0.9, remove_threshold=0.1):
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
        X_success, X_test_listing, filtered_features, successful,
        test_listing, top_k
    )

    recommendations = find_recommendations(
        add_threshold, current_features, filtered_features, remove_threshold,
        similar_features
    )

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
        in_bucket = False
        for pref in BUCKET_PREFIXES:
            if col.startswith(pref):
                in_bucket = True
                break
        if in_bucket:
            continue
        if current_features[col] < 0.5 and similar_features[col] >= add_threshold:
            recommendations[col] = "Consider ADDING (very common in successful listings)"
        elif current_features[col] > 0.5 and similar_features[col] <= remove_threshold:
            if not col.startswith(AMENITY_PREFIX):
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
    # Keep numeric features excluding success metrics
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

    # Fit scaler on train_df numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[numeric_features])

    # Transform successful and test listing
    X_success = scaler.transform(successful[numeric_features])
    X_test_listing, _ = preprocess_features(pd.DataFrame([test_listing_filled]), numeric_features, scaler)

    return X_success, X_test_listing, filtered_features, successful




def filter_successful_comparable_listings(success_metric, metric_threshold, test_listing, train_df):
    """
    Filter listings similar to the test listing AND successful by the chosen metric.
    success_metric: one of ["rating", "occupancy", "revenue"]
    threshold: percentile cutoff (e.g., 0.8 means top 20%)
    """
    metric_conf = SUCCESS_METRICS[success_metric]
    metric_col = metric_conf["col"]

    successful = filter_similar_pool(train_df, test_listing)

    if metric_col in successful.columns:
        cutoff = successful[metric_col].quantile(metric_threshold)
        successful = successful[successful[metric_col] >= cutoff]

    # Drop other success metric cols
    successful = successful.drop(columns=metric_conf["drop"], errors="ignore")
    test_listing = test_listing.drop(metric_conf["drop"], errors="ignore")

    return successful, test_listing


def normalize_metric_cols(df, metric_cols):
    # Normalize success metrics to 0-1
    for metric in metric_cols:
        if metric in df.columns:
            min_val = df[metric].min()
            max_val = df[metric].max()
            if max_val > min_val:  # avoid division by zero
                df[metric] = (df[metric] - min_val) / (max_val - min_val)
            else:
                df[metric] = 0.5  # fallback if all values are equal


# -----------------------------
# Example Pipeline
# -----------------------------
def run_pipeline(csv_path, random_state=42, success_metric="rating",
                 metric_threshold=0.9, top_k=20, add_threshold=0.80, remove_threshold=0.20,
                 test_rating_threshold=0.5):
    """
    success_metric: one of ["rating", "occupancy", "revenue"]
    threshold: percentile cutoff for defining success
    """
    df = pd.read_csv(csv_path)
    normalize_metric_cols(df, SUCCESS_METRIC_COLS)
    feature_cols = [
        col for col in df.columns
        if col not in IMMUTABLE_COLS
           and col not in EXCLUDE_FROM_RECOMMEND
           and col not in REMOVE_COLS
    ]

    success_metric_col, test_listing = sample_bad_listing(random_state, success_metric, df, test_rating_threshold)
    # Remove rating if present
    if success_metric_col in test_listing.index:
        print(success_metric, test_listing[success_metric_col])
        test_listing = test_listing.drop(success_metric_col)

    recs = recommend_features(
        df, test_listing, feature_cols, top_k=top_k,
        success_metric=success_metric, metric_threshold=metric_threshold,
        add_threshold=add_threshold, remove_threshold=remove_threshold
    )

    print("\nPicked test listing ID:", test_listing.get("id", "unknown"))

    return recs


def sample_bad_listing(random_state, success_metric, test_df, test_rating_threshold):
    success_metric_col = SUCCESS_METRICS[success_metric]["col"]
    # Pick test listing (for "rating" we still try to pick low-rated ones)
    if success_metric_col in test_df.columns:
        test_listing = filter_bad_listing(random_state, success_metric_col, test_df, test_rating_threshold)
    else:
        test_listing = test_df.sample(1, random_state=random_state).iloc[0]
    test_listing = test_listing.drop(columns=SUCCESS_METRICS[success_metric]["drop"])
    return success_metric_col, test_listing


def filter_bad_listing(random_state, success_metric_col, test_df, test_rating_threshold):
    bad_listings = test_df[test_df[success_metric_col] <= test_rating_threshold]
    if bad_listings.empty:
        print(
            f"No test listings below estimated revenue threshold {test_rating_threshold}. Using random test listing.")
        test_listing = test_df.sample(1, random_state=random_state).iloc[0]
    else:
        test_listing = bad_listings.sample(1, random_state=random_state).iloc[0]
    return test_listing

def analyze_all_metrics(csv_path, k_range=range(1, 51),
                        remove_threshold=0.30, add_threshold=0.70):
    all_results = []

    for metric in ["rating", "occupancy", "revenue"]:
        for k in tqdm(k_range, desc=f"Testing k for {metric}"):
            recs = run_pipeline(
                csv_path, top_k=k,
                remove_threshold=remove_threshold,
                add_threshold=add_threshold,
                success_metric=metric
            )

            n_add = sum("ADDING" in v for v in recs["recommendations"].values())
            n_remove = sum("REMOVING" in v for v in recs["recommendations"].values())
            n_bucket = len(recs["buckets"])
            avg_agreement = np.mean(list(recs["agreements"].values())) if recs["agreements"] else 0

            all_results.append({
                "metric": metric,
                "k": k,
                "add": n_add,
                "remove": n_remove,
                "bucket": n_bucket,
                "avg_agreement": avg_agreement
            })

    return pd.DataFrame(all_results)


def plot_all_metrics(df):
    # Melt the dataframe so we can map "type" (add/remove/bucket) separately
    df_melt = df.melt(id_vars=["metric", "k"],
                      value_vars=["add", "remove", "bucket"],
                      var_name="type", value_name="count")

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_melt, x="k", y="count",
                 hue="metric", style="type", linewidth=2, markers=True)
    plt.ylabel("Number of Recommendations")
    plt.title("Effect of k on Recommendations (all metrics)")
    plt.legend(title="Metric / Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Agreement plot stays simple
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x="k", y="avg_agreement", hue="metric", linewidth=2)
    plt.ylabel("Average Feature Agreement (%)")
    plt.title("Average Agreement vs k (all metrics)")
    plt.show()




if __name__ == '__main__':
    path = r"C:\Users\hodos\Documents\Uni\Uni-Year-3\Semester2\Data\final_norm_database.csv"
    # run_pipeline(path, top_k=25, remove_threshold=0.30, add_threshold=0.70, success_metric="revenue")
    df_all = analyze_all_metrics(path, k_range=range(1, 51))
    plot_all_metrics(df_all)