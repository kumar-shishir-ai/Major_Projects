import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
from PIL import Image
import base64
import json
import pathlib


model = load_model('emotion_detect_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}


st.set_page_config(page_title="Emotion Detection", page_icon="😊", layout="wide")

st.markdown("""
    <style>
        .title {
            color: white;
            font-size: 45px;
            text-align: center;
            font-weight: 700;
            padding:15px;
            background: linear-gradient(to right, rgb(29, 4, 108),rgb(120, 2, 63));
            border-radius:10px;
            }
            .para {
            text-align: center;
            font-size: 17px;
            font-weight:500;
        }
    </style>
""", unsafe_allow_html=True)


def load_lottie_file(filepath:str):
    with open(filepath,"r")as f:
        return json.load(f)


# st.title("😊 Real-Time Emotion Detection App")
st.markdown('<div class="title">😊 Real-Time Emotion Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="para">Detect emotions from Live Webcam or Uploaded Images using a CNN .</div>', unsafe_allow_html=True)
st.markdown("---")


# Function to Predict Emotion
def detect_emotion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        resized = cv2.resize(face, (48,48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1,48,48,1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        emotion = labels_dict[label]

        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return img

# Web Camp Input _________________
col = st.columns(2)
with col[0]:
    st.subheader("🎥 Live Emotion Detection")
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Camera not detected!")
            break
        frame = cv2.flip(frame, 1)
        frame = detect_emotion(frame)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    camera.release()
    # st.info("✅ Webcam Stopped.")
    st.markdown(
        '<div class="result-box" style="background-color:#262730; color:white;padding:10px;text-align:center;font-size:20px;font-weight:500;border-radius:10px;margin-top:-10px;">'
        '✅ Webcam Stopped.</div>',
        unsafe_allow_html=True)


# Image Input ____________________

with col[1]:
    st.subheader("🖼️ Upload an Image for Emotion Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_np = np.array(img)
        if img_np.shape[-1] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        result_img = detect_emotion(img_np)
        # st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detected Emotion", use_column_width=True)
        result_img = cv2.resize(result_img,(620,480))
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detected Emotion", use_column_width=True)


data_lottie = load_lottie_file("CCTV Camera.json")
st_lottie(
    data_lottie,
    height=300,
    width=None
)


