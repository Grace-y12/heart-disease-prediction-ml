# 💓 Heart Disease Prediction App

A Machine Learning project that predicts the likelihood of a person having heart disease based on clinical attributes. Built with `scikit-learn`, deployed using `Streamlit`, and powered by real-world health data.


## 🚀 Features

- Interactive web app built with **Streamlit**
- Clean interface for user input
- One-hot encoding, scaling, and preprocessing pipeline
- Trained using **Logistic Regression** (can be swapped for XGBoost or others)
- Displays clear prediction results: ✅ Low risk or ⚠️ High risk


## 🧠 Dataset

Uses the classic **Heart Disease UCI dataset**, which includes 13 features such as age, cholesterol, chest pain type, etc.

You can find the dataset on [Kaggle](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) or the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease).


## 📁 Project Structure
ML/

├── data

|   ├── heart-disease.csv

├── notebooks

|   ├──01_EDA.ipynb

├── app.py # Streamlit web app

├── src/

│ ├── train_model.py # Model training script

│ ├── scaler.pkl # StandardScaler from training

│ ├── heart_model.pkl # Trained model

│ ├── columns.pkl # Column order for input encoding

│ 
├── requirements.txt # Python dependencies

└── README.md # This file

#### 🛠️ Installation & Usage
```bash
1. Clone the repository:

git clone https://github.com/Grace-y12/heart-disease-prediction-ml.git
cd heart-disease-prediction-ml

2.Install dependencies:

pip install -r requirements.txt

3.Run the app:

streamlit run app.py
Open http://localhost:8501 in your browser.

4.🔍 Model Training (Optional)
To retrain the model with your own tweaks:

python src/train_model.py
This will output:

heart_model.pkl

scaler.pkl

columns.pkl
```

## 📷 App Preview
<img width="975" height="785" alt="image" src="https://github.com/user-attachments/assets/3df255ed-c3c7-4670-959d-d1c618f12b51" />

<img width="975" height="574" alt="image" src="https://github.com/user-attachments/assets/f8d256b6-ecc1-49e7-8ca7-91179f2d52ab" />


## 📦 Built With
Python

Pandas

Scikit-learn

Streamlit

Matplotlib & Seaborn

## 👩‍💻 Author
Grace Chundu
Computer Engineering Graduate | AI & Cybersecurity Intern
LinkedIn | GitHub




