import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns
import numpy as np
from sklearn.metrics import pairwise_distances


# --- PANDAS DISPLAY OPTIONS ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# ==============================================================================
# Main Clustering and Analysis Workflow
# ==============================================================================
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
# --- Compute silhouette score ---
sil_score = silhouette_score(X_binary, df["cluster"])
print(f"Silhouette Score for K-Means with k={best_k}: {sil_score:.3f}")

# --- 4. ANALYZE CLUSTERS ---
print(f"\n{'=' * 20} Cluster Analysis {'=' * 20}")

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
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=[f"Cluster {i}" for i in sorted(df["cluster"].unique())],
    title="Clusters"
)
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

# --- AVERAGE PRICE PER K-MEANS CLUSTER ---
if "price" in df.columns:
    avg_price_kmeans = df.groupby("cluster")["price"].mean().round(2)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=avg_price_kmeans.index, y=avg_price_kmeans.values, palette="viridis")
    plt.title("Average Price per K-Means Cluster", fontsize=16)
    plt.xlabel("Cluster", fontsize=12)
    plt.ylabel("Average Price ($)", fontsize=12)

    # Annotate bars with values
    for i, val in enumerate(avg_price_kmeans.values):
        plt.text(i, val, f"${val:.0f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
else:
    print("No 'price' column in dataframe — cannot compute average price per cluster.")


# --- 6. VISUALIZE SIMILARITY MATRIX ---
print("\nCreating Similarity Matrix Visualization...")

# First, order the data by cluster to group similar points together
df_sorted = df.sort_values(by="cluster").reset_index(drop=True)

# Sample a smaller subset to avoid long computation times on a large dataset
# and make the visualization clearer
# Sampling 200 listings from each cluster
sample_df = df_sorted.groupby('cluster', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 1000), random_state=42))

# Use the binary features for distance calculation
X_sample = sample_df[all_binary_cols]

# Calculate the pairwise similarity matrix (1 - distance)
# Using 'jaccard' is a good choice for binary data
similarity_matrix = 1 - pairwise_distances(X_sample.values, metric='jaccard')
# Create the heatmap plot
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, cmap='jet', vmin=0, vmax=1)
plt.title("Similarity Matrix of Sampled Listings")
plt.xlabel("Points")
plt.ylabel("Points")
plt.tight_layout()

plt.show()
