import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from src.utils.utils import render_navigation, config_html


class BatchPredictionApp:
    def __init__(self):
        st.set_page_config(page_title="Batch Prediction", layout="wide")
        render_navigation()
        config_html()
        st.title("Batch Prediction")

        self.model = st.session_state["model"]
        self.model_name = st.session_state["model_name"]
        self.exp_name = st.session_state["exp_name"]
        self.exp_path = os.path.join("experiments", self.exp_name)

        with open(os.path.join(self.exp_path, "config.json")) as f:
            self.config = json.load(f)

        self.dftest = []
        self.combined_dftest = None

    def upload_data(self):
        upload_files = st.file_uploader("Upload your test set", accept_multiple_files=True)
        if upload_files:
            for file in upload_files:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                    self.dftest.append(df)
                elif file.name.endswith(".txt"):
                    content = file.read().decode("utf-8", errors="ignore")
                    df = pd.DataFrame({"content": [content]})
                    self.dftest.append(df)

            self.combined_dftest = pd.concat(self.dftest, ignore_index=True)
            st.write("Uploaded data:")
            st.dataframe(self.combined_dftest.head())

    def make_predictions(self):
        if self.combined_dftest is not None and st.button(" 🚀 Make Prediction"):
            st.markdown("## Summary Statistics")
            preds = self.model.predict(self.combined_dftest[["content"]])
            probabilities = self.model.predict_proba(self.combined_dftest[["content"]])

            result = pd.DataFrame({
                "text": self.combined_dftest["content"],
                "predicted_label": preds,
                "probability": probabilities.max(axis=1),
                "predicted_class": probabilities.argmax(axis=1)
            })

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Distribution of Predicted Labels")
                fig, ax = plt.subplots()
                sns.countplot(data=result, x="predicted_label", ax=ax)
                ax.set_title("Distribution of Predicted Labels")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("Count")
                st.pyplot(fig)

            with col2:
                st.markdown("### Average Probability for Each Class")
                avg_prob = result.groupby("predicted_label")["probability"].mean().reset_index()
                fig, ax = plt.subplots()
                sns.barplot(data=avg_prob, x="predicted_label", y="probability", ax=ax)
                ax.set_title("Average Probability for Each Class")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("Average Probability")
                st.pyplot(fig)

    def run(self):
        self.upload_data()
        self.make_predictions()


if __name__ == "__main__":
    app = BatchPredictionApp()
    app.run()