import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

from preprocessing import load_and_preprocess

# Load and preprocess the data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
joblib.dump(scaler, "src/scaler.pkl")
X_train_df = pd.DataFrame(X_train)
joblib.dump(X_train_df.columns.tolist(), "src/columns.pkl")




models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    print(f"🔹 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    results[name] = acc

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name} with Accuracy: {results[best_model_name]:.3f}")

# Save model and scaler
joblib.dump(best_model, "src/heart_disease_model.pkl")
joblib.dump(scaler, "src/scaler.pkl")


from sklearn.model_selection import cross_val_score

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name} CV Mean Accuracy: {scores.mean():.3f}")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Predict
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]  # for ROC-AUC

# 1. Classification Report
print("📋 Classification Report:")
print(classification_report(y_test, y_pred))

# 2. Confusion Matrix
print("🧩 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# 3. ROC-AUC Curve
print("📈 ROC-AUC Score:", roc_auc_score(y_test, y_proba))
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend()
plt.grid(True)
plt.show()

