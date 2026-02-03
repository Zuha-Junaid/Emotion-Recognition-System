# 🧠 Facial Emotion Recognition AI

A Deep Learning project that classifies human facial expressions into 7 categories using a Custom CNN and a Streamlit dashboard.

## 🚀 Overview
This project uses a Custom Convolutional Neural Network (CNN) trained on the FER-2013 dataset. It features an interactive web dashboard for real-time emotion prediction.

### Key Features:
* **Custom CNN Architecture:** Optimized for speed and efficiency.
* **Streamlit Dashboard:** Easy-to-use interface for image uploads.
* **EDA Tools:** Built-in exploratory data analysis visualization.
* **Class Weighting:** Specifically tuned to handle dataset imbalances (Happy vs. Disgust).

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Frameworks:** TensorFlow, Keras
* **UI:** Streamlit
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, PIL

## 📂 Project Structure
```text
├── app.py                # Streamlit Web Dashboard
├── train_model.py        # Model Training Script
├── eda.py                # Exploratory Data Analysis Script
├── evaluate_model.py     # Model Evaluation (Confusion Matrix)
├── emotion_model.h5      # Trained Model (Not included in repo)
└── requirements.txt      # List of dependencies
⚙️ How to Run
Clone the repo:

Bash
git clone [https://github.com/YOUR_USERNAME/Emotion-Recognition-AI.git](https://github.com/YOUR_USERNAME/Emotion-Recognition-AI.git)
cd Emotion-Recognition-AI
Install Dependencies:

Bash
pip install -r requirements.txt
Run the Dashboard:

Bash
streamlit run app.py
📊 Results
The model achieves high accuracy by balancing feature extraction and computational load. For detailed metrics, refer to the "Model Performance" section of the dashboard.


---

## Part 3: Creating the `requirements.txt`
GitHub users need to know what to install. Run this command in your terminal to generate the file automatically:

`pip freeze > requirements.txt`

