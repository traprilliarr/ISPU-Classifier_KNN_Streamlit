import streamlit as st
import pandas as pd

# Set the page layout
st.set_page_config(page_title="ISPU KNN Classifier", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Classification", "Prediction"]
page = st.sidebar.radio("Go to", pages)

if page == "Classification":
    st.sidebar.success("You are on the Classification page")
    import classification
    classification.app()
elif page == "Prediction":
    st.sidebar.success("You are on the Prediction page")
    import pred
    pred.app()
