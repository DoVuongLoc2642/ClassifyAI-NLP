import json
import streamlit as st
import pandas as pd
import os
import sys
# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import render_navigation, config_html

st.set_page_config(page_title= "ClassifyAI", layout="wide")
render_navigation()
config_html()
st.title("🧠 ClassifyAI - NLP Classification")


EXP_DIR = os.path.join("../experiments")
os.makedirs(EXP_DIR, exist_ok=True)

st.header("Experiment Options")
mode = st.radio("Select Mode", ("Select Mode", "Create New Experiment", "Load Existing Experiment")) # radio button for mode selection


# LOAD EXISTING EXPERIMENT
if mode == "Load Existing Experiment":
    st.subheader("Load Experiment")

    exp_list = os.listdir(EXP_DIR)
    if not exp_list:
        st.warning("No experiments found. Please create a new experiment.")
    else:
        exp_name = st.selectbox(("Select Experiment"), exp_list)
        exp_path = os.path.join(EXP_DIR, exp_name)
        df = pd.read_csv(os.path.join(exp_path, "data.csv"))
        with open(os.path.join(exp_path, "config.json")) as f:
            config = json.load(f)

        st.success(f"Loaded experiment `{exp_name}` with {len(df)} samples.")
        st.write("Sample Data:")
        st.dataframe(df.head())

        # save sesstion state
        st.session_state["df"] = df
        st.session_state["config"] = config

        if st.button("Load Experiment and Analyze data"):
            st.session_state["exp_name"] = exp_name
            st.session_state["config"] = config
            # switch to data analysis page
            st.switch_page("pages/data-analysis.py")


# CREATE NEW EXPERIMENT
elif mode == "Create New Experiment":
    st.subheader("Create a New Experiment")

    # Experiment Name
    exp_name = st.text_input("Experiment Name", "ClassifyAI")
    # select number of label with scroll bar
    num_labels = st.number_input("Number of Labels", min_value=2, max_value=10, step=1)

    # Label configuration
    label_data = []
    for i in range(num_labels):
        st.markdown(f"---")
        label_name = st.text_input(f"Label {i+1} Name: ", key=f"label_name_{i}")
        upload_file = st.file_uploader(f"Upload CSV files for label {i+1} ", accept_multiple_files=True)
        if label_name and upload_file:
            for file in upload_file:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                    df['category'] = label_name
                    label_data.append(df) # list of dataframes
                elif file.name.endswith(".txt"):
                    content = file.read().decode("utf-8", errors = "ignore")
                    df = pd.DataFrame({"content": [content]})
                    df["category"] = label_name
                    label_data.append(df)

    # Combine and Save
    if st.button("Save Experiment"):
        combined_df = pd.concat(label_data, ignore_index=True)
        text_columns = [col for col in combined_df.columns if combined_df[col].dtype == 'object'] # list of text columns

        st.success(f"Experiment `{exp_name}` created with {len(combined_df)} samples.")

        # save experiment
        exp_dir = os.path.join("experiments", exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        combined_df.to_csv(os.path.join(exp_dir, "data.csv"), index=False)


        config = {
            "experiment_name": exp_name,
            "target_col": "category",
        }
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        st.success(f"Experiment `{exp_name}` saved successfully.")
        st.write("Sample Data:")
        st.dataframe(combined_df.head())

        # save session state
        st.session_state["df"] = combined_df
        st.session_state["config"] = config
        st.session_state["exp_name"] = exp_name
        #swtich to data analysis page
        st.switch_page("pages/data-analysis.py")
