# utils.py
import streamlit as st

def render_navigation():
    with st.sidebar:
        st.title("🔎 Navigation")
        st.page_link("app.py", label="💾 Main Menu")
        st.page_link("pages/data-analysis.py", label="📊 Experiments Analysis")
        st.page_link("pages/train-model.py", label="🧠 Model Training")
        st.page_link("pages/realtime-prediction.py", label="🚀 Real-time Prediction")
        st.page_link("pages/batch-prediction.py", label="📂 Batch Prediction")

def config_html():
    # Hide default sidebar text
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create dummy option in radio and hidden it
    st.markdown(
        """
    <style>
        div[role=radiogroup] label:first-of-type {
            visibility: hidden;
            height: 0px;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )