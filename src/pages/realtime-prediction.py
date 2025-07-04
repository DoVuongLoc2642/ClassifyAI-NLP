import streamlit as st
import os
import json
import seaborn as sns
from matplotlib import pyplot as plt
from src.utils.utils import render_navigation, config_html
import numpy as np
import pandas as pd
import re


class RealTimePredictionApp:
    def __init__(self):
        st.set_page_config(page_title="Real-time Prediction", layout="wide")
        render_navigation()
        config_html()
        st.title("🚀 Real-time Prediction")

        self.model = st.session_state["model"]
        self.model_name = st.session_state["model_name"]
        self.exp_name = st.session_state["exp_name"]
        self.exp_path = os.path.join("experiments", self.exp_name)
        self.df = st.session_state["df"]

        with open(os.path.join(self.exp_path, "config.json")) as f:
            self.config = json.load(f)

    def highlight_important_words(self, text: str, model, vectorizer, top_n: int = 5) -> str:
        df_input = pd.DataFrame([text], columns=["content"])
        X_vec = vectorizer.transform(df_input)

        tfidf = vectorizer.named_transformers_['content'].named_steps['tfidf']
        feature_names = tfidf.get_feature_names_out()

        X_array = X_vec.toarray()[0]

        if hasattr(model, "coef_"):
            pred_class = model.predict(X_vec)[0]
            class_index = list(model.classes_).index(pred_class)
            weights = model.coef_[class_index] if len(model.coef_.shape) > 1 else model.coef_[0]
            contrib = X_array * weights

        elif hasattr(model, "feature_importances_"):
            weights = model.feature_importances_
            contrib = X_array * weights
        else:
            return text

        top_indices = np.argsort(contrib)[-top_n:]
        top_words = [feature_names[i] for i in top_indices if contrib[i] > 0]

        highlighted_text = text
        for phrase in sorted(top_words, key=lambda x: -len(x)):
            pattern = re.compile(rf'(?i)\b{re.escape(phrase)}\b')
            highlighted_text = pattern.sub(
                rf"<mark style='background-color: #FFD54F'><b>{phrase}</b></mark>",
                highlighted_text
            )

        return highlighted_text

    def input_text_and_predict(self):
        text = st.text_input("Enter text for prediction", placeholder="Type here...")
        if text:
            df = pd.DataFrame([text], columns=["content"])
            st.write(df)

        if st.button("Predict") and text:
            st.markdown("## Summary Statistics")
            preds = self.model.predict(df[["content"]])
            pred_proba = self.model.predict_proba(df[["content"]])

            highlighted = self.highlight_important_words(
                text=text,
                model=self.model.named_steps["classifier"],
                vectorizer=self.model.named_steps["preprocessing_data"],
                top_n=50
            )

            st.markdown("### 🔍 Key Points in Text")
            st.markdown(highlighted, unsafe_allow_html=True)

            result = pd.DataFrame({
                "Text": text,
                "Predicted_label": preds,
                "Probability": pred_proba.max(axis=1)
            })

            st.markdown("### 🧠 Prediction Result: " + preds[0])
            st.dataframe(result)

            labels = self.model.classes_
            proba_df = pd.DataFrame(pred_proba, columns=labels)

            st.markdown("### 🎯 Prediction Probabilities Across Classes")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(
                x=proba_df.columns,
                y=proba_df.iloc[0],
                ax=ax,
                palette="Blues_d"
            )
            ax.set_xlabel("Class Label")
            ax.set_ylabel("Probability")
            ax.set_title("Probability for Each Class")
            st.pyplot(fig)

    def run(self):
        self.input_text_and_predict()


if __name__ == "__main__":
    app = RealTimePredictionApp()
    app.run()
