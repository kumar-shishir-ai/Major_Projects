import streamlit as st
import numpy as np
import pickle
import re
import json
import pathlib
import base64
from streamlit_lottie import st_lottie
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(
    page_title="📰 Fake News Buster",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_lottie_file(filepath:str):
    with open(filepath,"r")as f:
        return json.load(f)

def css_file(file_path):
    with open(file_path) as f:
        st.html(f"<style>{f.read()}</style>")

filepath = pathlib.Path("fake.css")
css_file(filepath)

def stemming(content):
    con = re.sub('[^\w\s]','',content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    return ' '.join(con)

def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction


st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 45px;
            font-weight: 800;
            background: linear-gradient(to right, white,blue);
            color: transparent;
            -webkit-background-clip: text;
            
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            margin-top: -10px;
            background: linear-gradient(to right, white,blue);
            color: transparent;
            -webkit-background-clip: text;
        }
        .stTextArea textarea {
            font-size: 16px !important;
        }
        .result-box {
            text-align: center;
            font-size: 25px;
            font-weight: 700;
            margin-top: 20px;
            border-radius: 12px;
            padding: 15px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Fake News Buster</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect whether a news article is Real or Fake instantly!</div>', unsafe_allow_html=True)

with st.sidebar:
    st.title("Fake News Buster 📰")
    st.markdown("---")
    st.markdown(
        """
        **Detect whether a piece of news is likely _real_ or _fake_.**

        **How to use**
        - Paste article text → Click **Predict**
        **Note:** This tool flags suspicious content — it's not a substitute for human fact-checking.
    """
     )

    data_lottie = load_lottie_file("Rocket.json")
    st_lottie(
        data_lottie,
        height=400,
        width=None
    )

news_text = st.text_area("📝 Enter the News Article or Headline:", height=200, placeholder="Type or paste your news text here...")
predict_btt = st.button("🚀 Analyze News",key="predict")
if predict_btt:
    prediction_class=fake_news(news_text)
    print(prediction_class)
    if prediction_class == [0]:
        st.markdown(
            '<div class="result-box" style="background: linear-gradient(to right, rgb(29, 4, 108),rgb(120, 2, 63)); color:white;">✅ This news appears to be REAL</div>',
            unsafe_allow_html=True)
    elif prediction_class == [1]:
        st.markdown(
            '<div class="result-box" style="background: linear-gradient(to right, rgb(4, 97, 123),rgb(4, 4, 134)); color:white;">🚫 This news appears to be FAKE</div>',
            unsafe_allow_html=True)

st.markdown("""
<hr>
<div style="text-align:center; font-size:15px; color:#7F8C8D;">
    Built with ❤️ using Streamlit | © 2025 Fake News Buster
</div>
""", unsafe_allow_html=True)