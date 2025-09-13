import sys
import streamlit as st 
import requests
import pandas as pd   
from movierecommender.exception.exception import MovieRecommenderException

API_URL = st.sidebar.text_input("API URL", "http://localhost:8000")

st.title("Movie Recommender - Interactive Dashboard")

user_id = st.number_input("User ID", min_value=1, value=1, step=1)
top_k = st.slider("Top K recommendations", 1, 50, 10)

if st.button("Get Recommendation"):
    payload = {"user_id": int(user_id), "top_k": int(top_k)}
    try:
        with st.spinner("Requesting Recommendations..."):
            resp = requests.post(f"{API_URL}/recommend", json=payload, timeout=300)
        if resp.status_code == 200:
            data = resp.json()
            recs = data["Recommendations"]
            df = pd.DataFrame(recs)
            st.success(f"Top {len(df)} recommendation for user { user_id}")
            st.table(df[['movieId', 'title', 'score']])
        else:
            st.error(f"API error: {resp.status_code} {resp.text}")
    except Exception as e:
        raise MovieRecommenderException(e, sys)