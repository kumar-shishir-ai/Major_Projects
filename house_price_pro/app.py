import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import pathlib
import pickle
import json
import os

st.set_page_config(
    page_title="Smart House Price Predictor",
    page_icon="🏡",
    layout="wide",
)

def load_lottie_file(filepath:str):
    with open(filepath,"r")as f:
        return json.load(f)



st.markdown("""
    <style>
        /* Title Animation */
        h1 {
            text-align: center;
            font-size: 2.1rem !important;
            color:white;
            background: linear-gradient(to right, rgb(29, 4, 108),rgb(120, 2, 63));
            font-weight: 800;
            border-radius:10px;
        }
        h5{
        /*color:white;
        text-align:center;
        background: linear-gradient(to right, rgb(4, 97, 123),rgb(4, 4, 134));
        padding:7px;
        border-radius:10px;
        align-item:center;
        display:flex;
        gap:10px;*/
        }
        p{
        text-align:center;
        color:white;
        align-item:center;;
    </style>
""", unsafe_allow_html=True)


def css_file(file_path):
    with open(file_path) as f:
        st.html(f"<style>{f.read()}</style>")

filepath = pathlib.Path("house.css")
css_file(filepath)



model = pickle.load(open("Ridge.pkl", "rb"))

st.markdown("<h1>🏡 Smart House Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p>Fill in the property details below to estimate the house price using the trained Regression model.</p>", unsafe_allow_html=True)

# ------------------------ SIDEBAR ------------------------
with st.sidebar:
    st.title("⚙️ App Info")
    st.markdown("<h5>✅Predicts house prices instantly based on user inputs</h5>", unsafe_allow_html=True)
    st.markdown("<h5>✅Dynamic location selection based on the dataset</h5>", unsafe_allow_html=True)
    st.markdown("<h5>✅Converts predicted prices into Lakhs (₹)</h5>", unsafe_allow_html=True)
    st.markdown("<h5>✅Interactive and animated Streamlit UI</h5>", unsafe_allow_html=True)
    data_lottie = load_lottie_file("analytics.json")
    st_lottie(
        data_lottie,
        height=300,
        width=None
    )


col = st.columns(2)
with col[0]:
    location = st.selectbox("**📍 Enter Location**",
                            ["Whitefield", "Sarjapur  Road", "Electronic City", "Raja Rajeshwari Nagar",
                             "Banjara Layout", "Vishwapriya Layout", "Thyagaraja Nagar", "Vishveshwarya Layout",
                             "Marsur", "Uttarahalli", "Haralur Road", "Marathahalli", "Bannerghatta Road",
                             "Hennur Road", "Thanisandra", "Hebbal", "Kanakpura Road", "Electronic City Phase II",
                             "Yelahanka", "7th Phase JP Nagar", "Bellandur", "KR Puram", "Chandapura", "Harlur",
                             "HAL 2nd Stage", "Laggere", "Mahalakshmi Layout", "Nagasandra", "Shampura",
                             "Poorna Pragna Layout",
                             "2nd Stage Nagarbhavi", "Doddakallasandra", "5th Block Hbr Layout", "Kodigehalli",
                             "Shivaji Nagar", "ITPL", "Giri Nagar", "Dasarahalli", "Konanakunte"])
    total_sqft = st.number_input("**📏 Total Area (sqft)**", min_value=200.0, max_value=30000.0, value=1200.0, step=50.0)
with col[1]:
    bhk = st.number_input("**🛏️ Bedrooms (BHK)**", min_value=1, max_value=16, value=3, step=1)
    bath = st.number_input("**🚿 Bathrooms**", min_value=1, max_value=16, value=2, step=1)

if st.button("**Predict**",key="predict"):
    features = pd.DataFrame(
        [[location, total_sqft, bath, bhk]],
        columns=["location", "total_sqft", "bath", "bhk"]
    )
    prediction = model.predict(features)
    st.subheader(f"💰 Estimated Price ₹: {prediction[0]*1e5:,.2f}")
