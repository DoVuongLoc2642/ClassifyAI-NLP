import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from src.utils.utils import render_navigation, config_html


class TrainModelApp:
    def __init__(self):
        st.set_page_config(page_title="Train Model", layout="wide")
        render_navigation()
        config_html()
        st.title("🚀 Train Model")

        self.df = st.session_state["df"]
        self.exp_name = st.session_state["exp_name"]
        self.exp_path = os.path.join("experiments", self.exp_name)
        with open(os.path.join(self.exp_path, "config.json")) as f:
            config = json.load(f)
        self.target_col = config["target_col"]

    def run(self):
        x = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col]

        st.markdown("### Select Test Size")
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

        st.markdown("### Select Model")
        model_name = st.radio("Model", ("Select Option", "Random Forest", "Logistic Regression", "SVM"), index=0)
        model = self.get_model(model_name)

        if "model_trained" not in st.session_state:
            st.session_state.model_trained = False

        if st.button(" 🚀 Train the Model"):
            regs = self.train_model(x_train, y_train, model)
            preds = regs.predict(x_test)
            report_df = self.plot_model_result(regs, x_test, y_test, labels=regs.classes_)

            st.session_state["model"] = regs
            st.session_state["model_name"] = model_name
            st.session_state.model_trained = True

        if st.session_state.model_trained:
            st.header("✅ Next Step: Real-time Prediction")
            if st.button("🤖Make Prediction"):
                st.switch_page("pages/realtime-prediction.py")
            elif st.button("📂 Batch Prediction"):
                st.switch_page("pages/batch-prediction.py")

    def get_model(self, model_name):
        model_map = {
            "Select Option": None,
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(),
            "SVM": SVC(probability=True)
        }
        return model_map[model_name]

    @staticmethod
    def preprocess_text(text_series):
        lemmatizer = WordNetLemmatizer()

        def clean(text):
            text = str(text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            words = text.lower().split()
            lemmatized = [lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized)

        return text_series.apply(clean)

    def train_model(self, x_train, y_train, model):
        preprocess_pipeline = Pipeline([
            ('preprocess', FunctionTransformer(self.preprocess_text, validate=False)),
            ('tfidf', TfidfVectorizer(
                stop_words=stopwords.words("english"),
                ngram_range=(1, 3),
                min_df=0.01,
                max_df=0.99,
                smooth_idf=True
            ))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('content', preprocess_pipeline, 'content'),
        ])

        regs = Pipeline([
            ('preprocessing_data', preprocessor),
            ('classifier', model)
        ])

        regs.fit(x_train, y_train)
        return regs

    def plot_model_result(self, model, x_test, y_true, labels=None):
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_true, y_pred)
        st.markdown(f"### ✅ Accuracy: `{acc:.4f}`")

        st.markdown("### 📉 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

        st.markdown("### 📊 Classification Report")
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df.style.highlight_max(axis=0, color="lightgreen"))

        st.markdown("### 📊 Classification Probability")
        probas = model.predict_proba(x_test)
        probas_df = pd.DataFrame(probas, columns=labels)
        probas_df = probas_df.mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        probas_df.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title("Average Classification Probability")
        ax.set_ylabel("Probability")
        ax.set_xlabel("Classes")
        ax.set_ylim(0, 1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

        return report_df


if __name__ == "__main__":
    app = TrainModelApp()
    app.run()
