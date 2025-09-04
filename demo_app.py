# demo_app.py
import streamlit as st
import pandas as pd
import numpy as np
from recommendation_system import run_pipeline, analyze_all_metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

# --------------------------
# Streamlit App
# --------------------------
st.title("🏠 Airbnb Listing Recommendation Interactive Demo")

# Upload CSV
csv_file = st.file_uploader("Upload Airbnb dataset (CSV)", type="csv")

if csv_file:
    df = pd.read_csv(csv_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # --------------------------
    # User controls
    # --------------------------
    success_metric = st.selectbox(
        "Choose success metric",
        ["rating", "occupancy", "revenue"]
    )
    top_k = st.slider("Number of neighbors (k)", 5, 100, 25)
    add_threshold = st.slider("Add threshold", 0.0, 1.0, 0.7)
    remove_threshold = st.slider("Remove threshold", 0.0, 1.0, 0.3)
    metric_threshold = st.slider("Percentile cutoff for success", 0.5, 1.0, 0.9)

    # Optionally pick a test listing or random
    listing_choice = st.selectbox(
        "Choose a listing by row index (or -1 for random)",
        [-1] + df.index.tolist()
    )

    # --------------------------
    # Prepare test listing
    # --------------------------
    if listing_choice == -1:
        test_listing = df.sample(1).iloc[0]
    else:
        test_listing = df.loc[listing_choice]

    st.subheader("Test Listing Preview")
    st.write(test_listing)

    # --------------------------
    # Run recommendation
    # --------------------------
    if st.button("Run Recommendation"):
        recs = run_pipeline(
            df,
            top_k=top_k,
            remove_threshold=remove_threshold,
            add_threshold=add_threshold,
            success_metric=success_metric,
            metric_threshold=metric_threshold
        )

        # ---- Recommendations ----
        st.subheader("🔎 Feature Recommendations")
        if recs["recommendations"]:
            st.dataframe(pd.DataFrame.from_dict(recs["recommendations"], orient="index", columns=["Suggestion"]))
        else:
            st.info("No feature recommendations for this listing.")

        # ---- Agreements ----
        st.subheader("📊 Feature Agreements with Neighbors")
        if recs["agreements"]:
            agreements_df = pd.DataFrame.from_dict(recs["agreements"], orient="index", columns=["Agreement %"])
            st.bar_chart(agreements_df.sort_values("Agreement %"))
        else:
            st.info("No agreements data available.")

        # ---- Bucket Recommendations ----
        st.subheader("📂 Bucket Recommendations")
        if recs["buckets"]:
            st.json(recs["buckets"])
        else:
            st.info("No bucket recommendations.")

        # --------------------------
        # Scatter Plot of Neighbors
        # --------------------------
        st.subheader("📍 Successful Neighbors vs Test Listing (2D PCA)")

        # Features for PCA
        IMMUTABLE_COLS = getattr(__import__('recommendation_system'), 'IMMUTABLE_COLS')
        EXCLUDE_FROM_RECOMMEND = getattr(__import__('recommendation_system'), 'EXCLUDE_FROM_RECOMMEND')
        REMOVE_COLS = getattr(__import__('recommendation_system'), 'REMOVE_COLS')
        feature_cols = [c for c in df.columns if c not in IMMUTABLE_COLS + EXCLUDE_FROM_RECOMMEND + REMOVE_COLS]

        # Fill NaNs
        df_features = df.copy()
        df_features[feature_cols] = df_features[feature_cols].fillna(0)
        test_vec = test_listing[feature_cols].fillna(0).values.reshape(1, -1)

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_features[feature_cols])
        X_test_scaled = scaler.transform(test_vec)

        # PCA to 2D
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
        X_test_2d = pca.transform(X_test_scaled)

        df_features['x'] = X_2d[:, 0]
        df_features['y'] = X_2d[:, 1]
        df_features['distance_to_test'] = np.linalg.norm(X_scaled - X_test_scaled, axis=1)

        # Find successful neighbors only
        success_metric_col = getattr(__import__('recommendation_system'), 'SUCCESS_METRICS')[success_metric]['col']
        cutoff = df_features[success_metric_col].quantile(metric_threshold)
        df_features['success'] = df_features[success_metric_col] >= cutoff

        # Sort by distance and take top_k successful neighbors
        neighbors_df = df_features[df_features['success']].nsmallest(top_k, 'distance_to_test')

        # Plot only neighbors + test listing
        fig = px.scatter(
            neighbors_df,
            x='x',
            y='y',
            color='distance_to_test',
            hover_data=neighbors_df.columns.tolist(),
            title='Successful Neighbors of Test Listing'
        )

        # Add the test listing
        fig.add_scatter(
            x=[X_test_2d[0, 0]],
            y=[X_test_2d[0, 1]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name='Test Listing'
        )

        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "Hover a neighbor to see its details and distance to the test listing. Red X = unsuccessful test listing.")
