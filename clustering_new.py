import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# --- PANDAS DISPLAY OPTIONS ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


# ==============================================================================
# Frequent Itemset & Association Rule Functions
# ==============================================================================

def find_top_associations(frequent_itemsets, num_rules=10):
    """Generates and prints the top association rules from frequent itemsets."""
    try:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        meaningful_rules = rules[
            (rules['support'] >= 0.05) &
            (rules['confidence'] >= 0.7) &
            (rules['lift'] > 1)
            ].copy()

        if meaningful_rules.empty:
            print("No meaningful association rules found with the current thresholds.")
            return

        meaningful_rules = meaningful_rules.sort_values(by='lift', ascending=False).head(num_rules)
        print("\n--- Top Association Rules ---")
        for _, row in meaningful_rules.iterrows():
            antecedent = ', '.join(list(row['antecedents']))
            consequent = ', '.join(list(row['consequents']))
            print(f"({antecedent}) → ({consequent}) | "
                  f"support={row['support']:.3f}, "
                  f"confidence={row['confidence']:.3f}, "
                  f"lift={row['lift']:.3f}")
    except Exception as e:
        print(f"Could not generate association rules: {e}")


def filter_columns_frequency(df, min_freq=0.05, max_freq=0.9):
    """Removes columns that are too common or too rare for meaningful analysis."""
    if df.empty:
        return df
    column_frequency = df.mean()
    keep_mask = (column_frequency >= min_freq) & (column_frequency <= max_freq)
    columns_to_keep = column_frequency[keep_mask].index
    return df[columns_to_keep]


def find_freq_itemsets(df):
    """Main function to run the frequent itemset mining process on a dataframe."""
    if df.empty:
        print("Dataframe is empty, skipping itemset analysis.")
        return

    df = df.dropna()
    df_filtered = filter_columns_frequency(df)

    if df_filtered.empty:
        print("No columns remaining after frequency filtering. Cannot find itemsets.")
        return

    try:
        frequent_itemsets = apriori(df_filtered, min_support=0.2, use_colnames=True, low_memory=True)
        print(f"Found {len(frequent_itemsets)} frequent itemsets.")

        if not frequent_itemsets.empty:
            find_top_associations(frequent_itemsets)

            large_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 4)]
            if not large_itemsets.empty:
                print("\n--- Top 5 Large Itemsets (by support) ---")
                print(large_itemsets.sort_values(by='support', ascending=False).head(5))
            else:
                print("No large itemsets (size > 2) found.")
        else:
            print("No frequent itemsets found with min_support=0.2.")

    except Exception as e:
        print(f"An error occurred during Apriori analysis: {e}")


# ==============================================================================
# Main Clustering and Analysis Workflow
# ==============================================================================

# Define the specific columns for frequent itemset analysis
# columns_for_frequent_itemsets = [
#     'superhost', 'host_profile_pic', 'host_verified', 'instant_bookable_bin',
#     'entire_house', 'shared_room_in_house', 'hotel/hostel_room',
#     'accommodates_1', 'accommodates_2', 'accommodates_3-4', 'accommodates_5-6',"accommodates_7+",
#     'bathrooms_0.5-1', 'bathrooms_1.5', 'bathrooms_2-2.5', 'bathrooms_3+',
#     'beds_0-1', 'beds_2', 'beds_3-4', 'beds_5+',
#     'price_100-160', 'price_160-256', 'price_256+',
#     'min_nights_1', 'min_nights_2', 'min_nights_3', 'min_nights_4+',
# ]


# --- 1. LOAD AND PREPARE DATA ---
try:
    df = pd.read_csv("final_norm_database.csv")
    print("Successfully loaded 'final_norm_database.csv'.")
except FileNotFoundError:
    print("Error: 'final_norm_database.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Define columns by type
numerical_cols = ['review_scores_rating', 'estimated_occupancy_l365d', 'estimated_revenue_l365d']
# All columns that are not numerical, id, or name are considered binary for clustering
all_binary_cols = [col for col in df.columns if col not in numerical_cols and col not in ['id', 'name']]

# Use ALL binary columns for clustering
X_binary = df[all_binary_cols].fillna(0)


# --- 2. ELBOW METHOD FOR OPTIMAL K ---
print("Running Elbow method to find optimal k...")
inertia = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_binary)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'o-', linewidth=2)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (Within-cluster SSE)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()

# --- 3. PERFORM CLUSTERING ---
best_k = 4
print(f"Performing K-Means clustering with k={best_k}...")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_binary)


# --- 4. ANALYZE CLUSTERS ---
print(f"\n{'=' * 20} Cluster Analysis {'=' * 20}")

# Part A: Analyze specific binary features with frequent itemset mining
cluster_labels = sorted(df["cluster"].unique())
for cluster_id in cluster_labels:
    print(f"\n{'=' * 25} Analysis for Cluster {cluster_id} {'=' * 25}")
    # Filter the DataFrame to the specific columns AND the current cluster
    try:
        cluster_df_filtered = df.loc[df["cluster"] == cluster_id, all_binary_cols].fillna(0)
        find_freq_itemsets(cluster_df_filtered)
    except KeyError as e:
        print(f"Error: One or more columns for frequent itemsets not found in DataFrame: {e}")
        continue

# Part B: Analyze numerical features by calculating averages
print(f"\n{'=' * 20} Numerical Feature Summary {'=' * 20}")
numerical_summary = df.groupby("cluster")[numerical_cols].mean()
print(numerical_summary)


# --- 5. VISUALIZE AND SAVE RESULTS ---
print(f"\n{'=' * 60}\nWorkflow Complete. Visualizing and saving final results...")

# PCA for visualization (still based on ALL binary columns used for clustering)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_binary)

plt.figure(figsize=(8, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["cluster"], cmap="viridis", s=40, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Airbnb Clusters (based on all binary features)")
plt.legend(handles=scatter.legend_elements()[0], labels=cluster_labels, title="Clusters")
plt.grid(True)
plt.show()

# Save the original dataframe with the new 'cluster' column
output_filename = "airbnb_clusters.csv"
df.to_csv(output_filename, index=False)
print(f"Clustering results saved to '{output_filename}'.")



# --- Binary Feature Summary per Cluster ---
print(f"\n{'=' * 20} Binary Feature Summary (percentage of 1s per cluster) {'=' * 20}")

binary_summary = df.groupby("cluster")[all_binary_cols].mean() * 100  # convert to %
binary_summary = binary_summary.round(1)  # round to 1 decimal place

print(binary_summary)

# (Optional) Save to CSV for easier inspection
binary_summary.to_csv("cluster_binary_summary.csv")
print("Binary feature summary saved to 'cluster_binary_summary.csv'.")

# --- 1. Compute summary statistics for clusters ---


# Assumes df has your data with a "cluster" column and binary feature columns (0/1)
cluster_summary = df.groupby("cluster").mean(numeric_only=True)

# --- 3. Cluster sizes ---
plt.figure(figsize=(6, 4))
df["cluster"].value_counts().sort_index().plot(kind="bar")
plt.title("Number of Listings per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# --- 4. Compare a few key features across clusters ---
key_features = ["accommodates_1","accommodates_2", "accommodates_3-4","accommodates_5-6", "accommodates_7+"]

cluster_summary[key_features].plot(kind="bar", figsize=(10, 6))
plt.title("Accommodates by Cluster (Proportions)")
plt.ylabel("Proportion (0-1)")
plt.xlabel("Cluster")
plt.legend(title="Features")
plt.tight_layout()
plt.show()


# --- 4. Compare a few key features across clusters ---
key_features = ['price_0-100', 'price_100-160', 'price_160-256', 'price_256+']
cluster_summary[key_features].plot(kind="bar", figsize=(10, 6))
plt.title("Price by Cluster (Proportions)")
plt.ylabel("Proportion (0-1)")
plt.xlabel("Cluster")
plt.legend(title="Features")
plt.tight_layout()
plt.show()


key_features = ['entire_house', 'shared_room_in_house', 'hotel/hostel_room']
cluster_summary[key_features].plot(kind="bar", figsize=(10, 6))
plt.title("Property Type by Cluster (Proportions)")
plt.ylabel("Proportion (0-1)")
plt.xlabel("Cluster")
plt.legend(title="Features")
plt.tight_layout()
plt.show()



