# ClassifyAI-NLP

ClassifyAI is a flexible and modular multi-model NLP platform for intelligent text classification. It supports a wide range of approaches—including traditional supervised learning allow users to experiment, compare, and deploy text classifiers with ease.

🧠 Built for researchers, developers, and data scientists, CategorizeAI includes:

- Unified interface for training, evaluating, and comparing multiple models

- Support for multi-class and multi-label classification tasks

- Real-time and batch prediction modes

- Interactive experiment tracking and visual analytics

Whether you're fine-tuning transformers or leveraging LLMs for reasoning-based classification, ClassifyAI provides the tools to accelerate your NLP workflows.

## ☀️ Key Features

- **Model Selection**: Select from a variety of Machine Learning models tailored to your domain, technical requirements, and available data.
- **Supervised Classification**: Provides an end-to-end workflow covering experiment setup, analysis, model training, and making predictions.
- **Experiment Management**: Easily save, load, and monitor various classification experiments, with flexible configurations to ensure reproducibility.
- **Responsive Design**: Features a contemporary, mobile-friendly interface with smooth animations and user-friendly navigation.

## 📋 Prerequisites

- Python 3.10 or higher
- Streamlit
- 8GB+ RAM 


## ⚙️ Installation & Setup

### Step 1: Set up the Environment

- Use Python version **3.10**
- It’s recommended to use conda to create the environment:

```
conda create -n myenv python=3.12.2
conda activate myenv
```

- Install requirements:

```
pip install -r requirements.txt
```

---

### Step 2: **Start the Application**

- Navigate to the `src` folder and run:

```
cd src
streamlit run navigation.py
```

## 💻 User Manual

### 🏠 Menu

The **Menu** page serves as the central hub of the **ClassifyAI** platform. From here, you can:

* 🔍 **Explore Features**: showcase different features offered for your use at the side bar: 💾 Loading Data Menu, 📊 Experiments Analysis, 🧠 Model Training, 🚀 Real-time Prediction, 📂 Batch Prediction
* 💽 **Manage Experiments**: Users can select to create new experiments or select existed experiments with its dataset and trained models. 
* 📁 **Upload Your Dataset**: Easily import CSV or text files with its label to begin trainning model in following step.
* 📊 **View Results & Visualizations**: Access and visualize dataset to gain meaningful insight of the dataset like: Top-Ngrams Plot, Class_distribution Plot, Word Clouds Plot,...
* ⚙️ **Configure Settings**: Customize test size split, select models and classification preferences and train model to reiceive its performance metric plot.
* 🚀 **Making Prediction**: Using model to predict a single new text or multiple text files from a folder. 
* 📌 **Navigate to Other Sections**: Quickly switch between training, predictions, and analytics dashboards.

The Menu provides an intuitive starting point for launching and managing your NLP experiments efficiently.

### 🎯 Supervised Classification

Train custom models using your labeled data—perfect for domain-specific tasks, high efficiency, or datasets with clear patterns! 🚀🤖

1. **Experiment Data Analysis** 📊  
   • Upload and configure experiments with your unique context and labeled classes.  
   • Get detailed statistical and distributional insights.  
   • Visualize lexical patterns, word frequencies, and semantic clusters for deeper understanding.

2. **Model Training** 🏋️‍♂️  
   • Choose your data and set up train/test splits.  
   • Benefit from automated text preprocessing.  
   • Track progress with intuitive visualizations: see classification reports and key word features.

3. **Real-Time Prediction** ⚡  
   • Instantly classify text with any trained model.  
   • Just enter your text and get immediate results!  
   • View probability distributions for all possible categories.

4. **Batch Prediction** 📂  
   • Classify multiple text files at once using trained models.  
   • Check out prediction distributions and confidence scores for each file.  
   • Export detailed results—including probability scores for every category.

---

## 📄 License

Distributed under the MIT License. 
