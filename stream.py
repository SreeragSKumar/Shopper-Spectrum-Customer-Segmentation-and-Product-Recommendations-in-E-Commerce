
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --------------------------------------
# Set Page Configuration
# --------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# --------------------------------------
# Load Models and Data
# --------------------------------------
kmeans = joblib.load("C:\\Users\\sreer\\OneDrive\\Desktop\\Shopper_Spectrum\\models\\kmeans_model.pkl")
scaler = joblib.load("C:\\Users\\sreer\\OneDrive\\Desktop\\Shopper_Spectrum\\models\\scaler.pkl")
similarity_df = joblib.load("C:\\Users\\sreer\\OneDrive\\Desktop\\Shopper_Spectrum\\models\\product_similarity.pkl")

# --------------------------------------
# Sidebar Navigation
# --------------------------------------
st.sidebar.title("🛒 Shopper Spectrum")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Customer Segmentation",
        "Product Recommendation"
    ]
)

# --------------------------------------
# Page: Home
# --------------------------------------
if page == "Home":

    st.title("🛒 Shopper Spectrum")

    st.write("""
    Customer Segmentation and Product Recommendation System

    Features:
    - Customer Segmentation using KMeans
    - Product Recommendation using Collaborative Filtering
    """)

# --------------------------------------
# Page: Customer Segmentation
# --------------------------------------
elif page == "Customer Segmentation":

    st.title("🎯 Customer Segmentation")

    recency = st.number_input(
        "Recency (days)",
        min_value=0
    )

    frequency = st.number_input(
        "Frequency",
        min_value=0
    )

    monetary = st.number_input(
        "Monetary",
        min_value=0.0
    )

    if st.button("Predict Segment"):

        input_data = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=[
                "Recency",
                "Frequency",
                "Monetary"
            ]
        )

        scaled_input = scaler.transform(input_data)

        cluster = kmeans.predict(scaled_input)[0]

        cluster_names = {
            0: "Occasional",
            1: "At Risk",
            2: "High Value",
            3: "Regular"
        }

        st.success(
            f"Customer Segment: {cluster_names[cluster]}"
        )

# --------------------------------------
# Page: Product Recommendation
# --------------------------------------
elif page == "Product Recommendation":

    st.title("📦 Product Recommender")

    product_name = st.text_input(
        "Enter Product Name"
    )

    if st.button("Get Recommendations"):

        product_name = product_name.upper().strip()

        if product_name not in similarity_df.columns:

            st.error("Product Not Found")

        else:

            recommendations = (
                similarity_df[product_name]
                .sort_values(ascending=False)
                .iloc[1:6]
                .index
                .tolist()
            )

            st.subheader(
                "Recommended Products"
            )

            for item in recommendations:
                st.write(f"✅ {item}")


