import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_preprocess(path="data/heart-disease.csv"):
    df = pd.read_csv(path)
    
    # Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # One-hot encoding
    X = pd.get_dummies(X, columns=['cp', 'thal', 'slope'])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler

def feature_engineering(df):
    # Age group
    bins = [0, 40, 55, 70, 100]
    labels = ['Young', 'Middle-Aged', 'Senior', 'Old']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

    # Interactions
    df['age_chol'] = df['age'] * df['chol']
    df['thalach_oldpeak'] = df['thalach'] * df['oldpeak']

    # Log transformation
    df['chol_log'] = np.log1p(df['chol'])
    df['oldpeak_log'] = np.log1p(df['oldpeak'])

    return df

df = pd.read_csv("data/heart-disease.csv")
df = feature_engineering(df)
df = pd.get_dummies(df, columns=['cp', 'thal', 'slope', 'age_group'])

