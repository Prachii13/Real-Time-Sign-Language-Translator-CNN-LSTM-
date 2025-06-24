import streamlit as st
import numpy as np
import tensorflow as tf

st.title("🤟 Sign Language Translator")

model = tf.keras.models.load_model("model.h5")

uploaded = st.file_uploader("Upload 30-frame video (64x64)", type=["mp4", "avi"])

if uploaded:
    st.video(uploaded)
    st.info("⚠️ Frame extraction from video not shown (needs OpenCV integration).")
    st.success("🤖 Predicted Sign: HELLO (demo output)")
