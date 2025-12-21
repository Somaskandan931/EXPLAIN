import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.title("Explainable Fake News Detection")

text = st.text_area("Enter news article")
model = st.selectbox("Select Model", ["xlmr", "indicbert", "tfidf"])

if st.button("Analyze"):
    response = requests.post(API_URL, json={
        "text": text,
        "model": model
    })
    st.json(response.json())
