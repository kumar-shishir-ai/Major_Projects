import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit_lottie import st_lottie
import pathlib
import base64
import json

model = pickle.load(open("kmeans.pkl","rb"))

st.set_page_config(page_title="", layout="wide")
st.markdown("""
<h1 style="color:black;text-align:center;">🧩 Customer Segmentation using K-Means Clustering</h1>
""",unsafe_allow_html=True)
st.info("This app helps businesses **understand their customers better** by grouping them into distinct segments based on their **age, income, spending habits, and gender**. By identifying which cluster a customer belongs.")
# --- User Inputs ---
def css_file(file_path):
    with open(file_path) as f:
        st.html(f"<style>{f.read()}</style>")

filepath = pathlib.Path("loan.css")
css_file(filepath)
col = st.columns(2)
with col[0]:
    def load_lottie_file(filepath: str):
        with open(filepath, "rb") as f:
            return json.load(f)


    data_lottie = load_lottie_file("filter.json")
    st_lottie(data_lottie, height=500, width=None)

with col[1]:
    st.markdown("""
    <h2 style="color:black;text-align:center;border-bottom:2px solid black;font-weight:600;">User Inputs</h2>
    """,unsafe_allow_html=True)
    gender = st.selectbox("**Select Gender:**", ["Male", "Female"])
    age = st.slider("**Select Age:**", 18, 70, 30)
    income = st.number_input("**Annual Income (k$):**", min_value=10, max_value=150, value=60)
    spending = st.slider("**Spending Score (1–100):**", 1, 100, 50)


    if st.button("Predict Cluster",key="predict"):
        user_data = pd.DataFrame([[gender, age, income, spending]],
                             columns=["Gender","Age","Annual Income (k$)","Spending Score (1-100)"])

        cluster = model.predict(user_data)[0]
        st.success(f"✅ This customer belongs to **Cluster {cluster}**")
        # Optional: Add some business insight messages
        cluster_messages = {
        0: "🟢 High spenders with moderate income.",
        1: "🔵 Low spenders — may need promotional offers.",
        2: "🟣 Young customers with varying habits.",
        3: "🟠 Loyal, steady buyers.",
        4: "🟤 Premium customers with high spending scores."
        }
        st.info(cluster_messages.get(cluster, "Customer profile not defined."))