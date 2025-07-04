import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import json
import os
from src.utils.utils import render_navigation, config_html


class DataAnalysisApp:
    def __init__(self):
        st.set_page_config(page_title="Experiments Analysis", layout="wide")
        render_navigation()
        config_html()

        self.df = st.session_state["df"]
        self.exp_name = st.session_state["exp_name"]
        self.exp_path = os.path.join("experiments", self.exp_name)

        with open(os.path.join(self.exp_path, "config.json")) as f:
            config = json.load(f)
        self.target_col = config["target_col"]

    def run(self):
        st.title("📊 Data Analysis")
        self.plot_ngrams_per_class()
        self.plot_class_distribution()
        self.plot_word_clouds()
        self.next_step()

    @staticmethod
    def get_top_ngrams(corpus, ngram_range=(1, 1), top_n=10):
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
        X = vectorizer.fit_transform(corpus)
        sum_words = X.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        return sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]

    def plot_ngrams_per_class(self, top_n=10):
        st.subheader("📊 N-gram Frequency Analysis")
        classes = self.df[self.target_col].unique()

        for cls in classes:
            st.markdown(f"### N-gram Frequency for Class: `{cls}`")
            class_texts = self.df[self.df[self.target_col] == cls]["content"].dropna().astype(str)

            fig, axes = plt.subplots(1, 3, figsize=(18, 4))

            for i, n in enumerate([1, 2, 3]):
                ngrams = self.get_top_ngrams(class_texts, ngram_range=(n, n), top_n=top_n)
                ngram_df = pd.DataFrame(ngrams, columns=["N-gram", "Frequency"])

                sns.barplot(data=ngram_df, x="Frequency", y="N-gram", ax=axes[i], palette="Set2")
                axes[i].set_title(f"Top {top_n} {n}-grams", fontsize=12)
                axes[i].set_xlabel("Frequency")
                axes[i].set_ylabel("")

            plt.tight_layout()
            st.pyplot(fig)

    def plot_class_distribution(self):
        st.subheader("Class Distribution Overview")
        class_counts = self.df[self.target_col].value_counts()
        labels = class_counts.index
        sizes = class_counts.values

        st.markdown("#### Pie Chart")
        colors = plt.get_cmap("Set2").colors[:len(labels)]

        fig, ax = plt.subplots(figsize=(8, 4))
        wedges, texts, autotexts = ax.pie(
            sizes,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops=dict(color="white", fontsize=10),
            wedgeprops=dict(edgecolor='white')
        )

        ax.axis("equal")
        ax.set_title("Class Distribution", fontsize=12)

        legend_labels = [f"{label} ({count})" for label, count in zip(labels, sizes)]
        ax.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
        st.pyplot(fig)

    def plot_word_clouds(self):
        def get_most_common_words(text, n=10):
            words = text.split()
            word_counts = pd.Series(words).value_counts()
            return word_counts.head(n)

        for cls in self.df[self.target_col].unique():
            col1, col2 = st.columns([2, 2])

            with col1:
                st.markdown("#### Word Cloud")
                wc = WordCloud(width=800, height=400, background_color="white").generate(
                    " ".join(self.df[self.df[self.target_col] == cls]["content"]))
                st.image(wc.to_array(), use_container_width=True)

            with col2:
                st.markdown("#### Most Common Words")
                common_words = get_most_common_words(" ".join(self.df[self.df[self.target_col] == cls]["content"]))
                st.bar_chart(common_words)

    def next_step(self):
        st.header("✅ Next Step: Train Your Model")
        if st.button("🤖Train Model"):
            st.switch_page("pages/train-model.py")


if __name__ == "__main__":
    app = DataAnalysisApp()
    app.run()
