# demo_app.py
import streamlit as st
import pandas as pd
import numpy as np
from recommendation_system import run_pipeline, IMMUTABLE_COLS
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

    # ✅ Threshold for picking the bad test listing
    test_listing_threshold = st.slider(
        "Threshold for test listing metric (bad listing cutoff)",
        0.0, 1.0, 0.5
    )

    # --------------------------
    # Run recommendation
    # --------------------------
    if st.button("Run Recommendation"):
        recs, test_listing = run_pipeline(
            df,
            top_k=top_k,
            remove_threshold=remove_threshold,
            add_threshold=add_threshold,
            success_metric=success_metric,
            metric_threshold=metric_threshold,
            test_rating_threshold=test_listing_threshold   # 👈 NEW
        )

        # ---- Show test listing ----
        st.subheader("📌 Test Listing Used")
        st.write(test_listing)

        st.subheader("Immutable Columns for Test Listing")

        immutable_values = test_listing[IMMUTABLE_COLS]
        # Filter to only the ones that are 1
        immutable_on = immutable_values[immutable_values == 1].index.tolist()

        if immutable_on:
            st.write(immutable_on)
        else:
            st.write("No immutable columns with value 1 for this listing.")

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



