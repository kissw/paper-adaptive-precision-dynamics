import streamlit as st

st.set_page_config(page_title="Deep Active Inference Dashboard", layout="wide")
st.title("Deep Active Inference — Autonomous Driving")
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", ["Training", "Evaluation", "Visualization"])

if page == "Training":
    from active_inference.ui.pages.training import render

    render()
elif page == "Evaluation":
    from active_inference.ui.pages.evaluation import render

    render()
elif page == "Visualization":
    from active_inference.ui.pages.visualization import render

    render()
