# demo_app.py
"""
Streamlit-based interactive demo for the Airbnb Listing Recommendation System.

This app allows users to:
1. Upload an Airbnb dataset in CSV format.
2. Preview the dataset.
3. Configure parameters for the recommendation system, including:
   - Success metric to optimize (rating, occupancy, revenue).
   - Number of neighbors (k).
   - Add/remove thresholds for feature recommendations.
   - Percentile cutoff for success.
   - Threshold for selecting a "bad" test listing.
4. Run the recommendation pipeline and view:
   - The chosen test listing.
   - Immutable columns for the listing.
   - Feature recommendations.
   - Agreements with neighbors.
   - Bucket-based recommendations.
"""

import streamlit as st
import pandas as pd
from recommendation_system import run_pipeline, IMMUTABLE_COLS


# --------------------------
# Streamlit App Title
# --------------------------
st.title("Airbnb Listing Recommendation Interactive Demo")

# --------------------------
# File Upload Section
# --------------------------
csv_file = st.file_uploader("Upload Airbnb dataset (CSV)", type="csv")

if csv_file:
    # Load the dataset
    df = pd.read_csv(csv_file)

    # Show preview of the dataset
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # --------------------------
    # User Controls
    # --------------------------
    # Select success metric
    success_metric = st.selectbox(
        "Choose success metric",
        ["rating", "occupancy", "revenue"]
    )

    # Configure thresholds and parameters
    top_k = st.slider("Number of neighbors (k)", 5, 100, 25)
    add_threshold = st.slider("Add threshold", 0.0, 1.0, 0.7)
    remove_threshold = st.slider("Remove threshold", 0.0, 1.0, 0.3)
    metric_threshold = st.slider("Percentile cutoff for success", 0.5, 1.0, 0.9)

    # Threshold for picking the bad test listing
    test_listing_threshold = st.slider(
        "Threshold for test listing metric (bad listing cutoff)",
        0.0, 1.0, 0.5
    )

    # --------------------------
    # Run Recommendation
    # --------------------------
    if st.button("Run Recommendation"):
        recs, test_listing = run_pipeline(
            df,
            top_k=top_k,
            remove_threshold=remove_threshold,
            add_threshold=add_threshold,
            success_metric=success_metric,
            metric_threshold=metric_threshold,
            test_rating_threshold=test_listing_threshold
        )

        # ---- Show test listing ----
        st.subheader("Test Listing Used")
        st.write(test_listing)

        # Show immutable columns for the test listing
        st.subheader("Immutable Columns for Test Listing")
        immutable_values = test_listing[IMMUTABLE_COLS]

        # Filter immutable columns to only the ones with value 1
        immutable_on = immutable_values[immutable_values == 1].index.tolist()

        if immutable_on:
            st.write(immutable_on)
        else:
            st.write("No immutable columns with value 1 for this listing.")

        # ---- Recommendations ----
        st.subheader("Feature Recommendations")
        if recs["recommendations"]:
            st.dataframe(pd.DataFrame.from_dict(
                recs["recommendations"], orient="index", columns=["Suggestion"]
            ))
        else:
            st.info("No feature recommendations for this listing.")

        # ---- Agreements ----
        st.subheader("Feature Agreements with Neighbors")
        if recs["agreements"]:
            agreements_df = pd.DataFrame.from_dict(
                recs["agreements"], orient="index", columns=["Agreement %"]
            )
            st.bar_chart(agreements_df.sort_values("Agreement %"))
        else:
            st.info("No agreements data available.")

        # ---- Bucket Recommendations ----
        st.subheader("Bucket Recommendations")
        if recs["buckets"]:
            st.json(recs["buckets"])
        else:
            st.info("No bucket recommendations.")
