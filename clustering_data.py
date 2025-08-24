
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns


def create_categorical_plots(df, feature, filename_prefix='categorical_plot'):
    """
    Generates and saves bar charts for a given categorical feature,
    showing its distribution within each cluster.
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='cluster', hue=feature, palette='viridis')
    plt.title(f'Distribution of {feature} by Cluster', fontsize=16)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title=feature)
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_{feature}.png')
    plt.close()
    print(f"Plot for {feature} saved as '{filename_prefix}_{feature}.png'")


def create_numerical_plots(df, feature, filename_prefix='numerical_plot'):
    """
    Generates and saves box plots for a given numerical feature,
    showing its distribution within each cluster.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y=feature, data=df, palette='viridis')
    plt.title(f'Distribution of {feature} by Cluster', fontsize=16)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_{feature}.png')
    plt.close()
    print(f"Plot for {feature} saved as '{filename_prefix}_{feature}.png'")


# Load your data
try:
    df = pd.read_csv("listings.csv")
except FileNotFoundError:
    print("The file 'listings.csv' was not found. Please make sure it's in the same directory.")
    exit()

# Remove property types with less than 100 listings
property_counts = df['property_type'].value_counts(dropna=False)
valid_property_types = property_counts[property_counts >= 100].index
df = df[df['property_type'].isin(valid_property_types)].copy()

# Keep all the original needed columns
cols_to_keep = [
    "host_response_rate",
    "neighbourhood_cleansed",
    "property_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "price",
    "instant_bookable",
    "review_scores_rating",
    "host_has_profile_pic"
]
df = df[cols_to_keep].copy()

# --- DATA CLEANING AND IMPUTATION ---
df["host_response_rate"] = (
    df["host_response_rate"]
    .str.replace("%", "", regex=False)
    .astype(float)
    / 100
)
df["instant_bookable"] = df["instant_bookable"].map({"t": 1, "f": 0}).fillna(0)
df["host_has_profile_pic"] = df["host_has_profile_pic"].map({"t": 1, "f": 0}).fillna(0)
df["price"] = df["price"].replace(r'[\$,]', '', regex=True).astype(float)

for col in ["bathrooms", "bedrooms", "beds"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

numeric_cols = [
    "host_response_rate", "accommodates", "bathrooms",
    "bedrooms", "beds", "price", "review_scores_rating"
]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = ["neighbourhood_cleansed", "property_type"]
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

price_95th_percentile = df["price"].quantile(0.95)
df["price_capped"] = df["price"].apply(lambda x: price_95th_percentile if x > price_95th_percentile else x)

df_encoded = pd.get_dummies(df, columns=["neighbourhood_cleansed", "property_type"], drop_first=True)
df_encoded = df_encoded.fillna(0)

features_for_clustering = [
    "host_response_rate", "accommodates", "bathrooms", "bedrooms",
    "beds", "price_capped", "instant_bookable", "host_has_profile_pic"
]
X = df_encoded[features_for_clustering]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- K-MEANS CLUSTERING ---
print("\n" + "="*50)
print("  ANALYSIS FOR K-MEANS CLUSTERING (k=6)")
print("="*50)

k = 6
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_clusters = kmeans.fit_predict(X_scaled)
df['kmeans_cluster'] = kmeans_clusters
df_encoded['kmeans_cluster'] = kmeans_clusters

print("\n--- K-Means Additional Insights ---")
silhouette_avg = silhouette_score(X_scaled, kmeans_clusters)
print(f"1. Silhouette Score: {silhouette_avg:.2f}")

print("\n2. K-Means Cluster Sizes:")
cluster_sizes = df['kmeans_cluster'].value_counts().sort_index()
print(cluster_sizes.to_string())

print("\n3. K-Means Cluster Feature Summary:")
cluster_summary = df.groupby('kmeans_cluster')[features_for_clustering + ['review_scores_rating']].mean().round(2)
print(cluster_summary.to_string())

# --- AGGLOMERATIVE CLUSTERING (DENDROGRAM) ---
print("\n" + "="*50)
print("  ANALYSIS FOR AGGLOMERATIVE CLUSTERING")
print("="*50)
print("Generating a dendrogram to help determine the optimal number of clusters.")

# Compute the linkage matrix
Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(15, 7))
dendrogram(
    Z,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90.,
    leaf_font_size=12.,
)
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Cluster Size or Original Data Point Index')
plt.ylabel('Distance')
plt.axhline(y=100, color='r', linestyle='--')
plt.savefig('dendrogram.png')
plt.close()
print("Dendrogram plot saved as 'dendrogram.png'")

print("\n--- Next Steps ---")
print("Examine the dendrogram to identify where a horizontal line would cross the most vertical lines.")
print("The number of vertical lines crossed will give you the optimal number of clusters.")
print("Once you've chosen a number, you can uncomment the Agglomerative Clustering section and set `n_clusters` to your chosen value.")
print("Then, re-run the code to see the results for your chosen number of clusters.")

# --- AGGLOMERATIVE CLUSTERING ANALYSIS (UNCOMMENT TO RUN) ---
# Once you have decided on the number of clusters (e.g., from the dendrogram),
# you can uncomment this section to run the analysis with that number.

# Set the number of clusters based on your dendrogram analysis
k_agg = 6

# Perform Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=k_agg)
agg_clusters = agg_clustering.fit_predict(X_scaled)
df['agg_cluster'] = agg_clusters
df_encoded['agg_cluster'] = agg_clusters

# Print analysis
print("\n--- Agglomerative Additional Insights ---")
silhouette_avg_agg = silhouette_score(X_scaled, agg_clusters)
print(f"1. Silhouette Score: {silhouette_avg_agg:.2f}")

print("\n2. Agglomerative Cluster Sizes:")
cluster_sizes_agg = df['agg_cluster'].value_counts().sort_index()
print(cluster_sizes_agg.to_string())

print("\n3. Agglomerative Cluster Feature Summary:")
cluster_summary_agg = df.groupby('agg_cluster')[features_for_clustering + ['review_scores_rating']].mean().round(2)
print(cluster_summary_agg.to_string())

# Generate plots for Agglomerative clusters
df['cluster'] = df['agg_cluster']
for feature in ["price", "accommodates", "bedrooms", "review_scores_rating"]:
    create_numerical_plots(df, feature, filename_prefix='agg_numerical')
for feature in ["neighbourhood_cleansed", "property_type"]:
    create_categorical_plots(df, feature, filename_prefix='agg_categorical')

# PCA visualization for Agglomerative clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
colors = cm.get_cmap('tab10', k_agg)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['agg_cluster'], cmap=colors, alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Agglomerative Clusters Visualized with PCA')
plt.colorbar(label='Cluster')
plt.savefig('agg_pca_clusters.png')
plt.close()
print("Agglomerative PCA visualization plot saved as 'agg_pca_clusters.png'")


# --- SAVE PROCESSED DATA ---
df_encoded.to_csv('preprocessed_listings_encoded.csv', index=False)
print("\nFinal preprocessed DataFrame saved as 'preprocessed_listings_encoded.csv'")
