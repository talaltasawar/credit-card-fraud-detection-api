# 💳 Credit Card Fraud Detection API

A **production-ready** Machine Learning REST API built with **FastAPI** that detects fraudulent credit card transactions in real-time. This project handles highly imbalanced data (0.17% fraud rate) using SMOTE and custom threshold tuning to achieve a strong recall of **91%**.

🔗 **Live Demo**: (Coming Soon on Render/Hugging Face)

---

## 📌 Overview

Credit card fraud is a critical problem for financial institutions. This API takes transaction features (anonymized via PCA) and returns a fraud probability score, risk level, and a clear prediction. 

The model is trained on the famous [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## 🛠️ Tech Stack

| Category | Technology |
| :--- | :--- |
| **API Framework** | FastAPI, Uvicorn |
| **Machine Learning** | Scikit-learn (Logistic Regression, SMOTE, StandardScaler) |
| **Data Handling** | Pandas, NumPy |
| **Serialization** | Joblib |
| **Validation** | Pydantic |
| **Language** | Python 3.13 |

---

## ✨ Key Features

- **Real-time Inference**: Predict fraud probability in milliseconds.
- **Imbalanced Data Handling**: Used **SMOTE** (Synthetic Minority Oversampling) to balance the training set.
- **Custom Threshold Tuning**: Optimized decision threshold at **0.85** to balance Precision and Recall.
- **Interactive Docs**: Auto-generated Swagger UI (`/docs`) and ReDoc (`/redoc`).
- **Beautiful UI**: Custom dark-theme homepage with glassmorphism effects.
- **Model Performance**:
  - **Recall**: ~91% (Catches most frauds)
  - **Precision**: ~14% (Due to extreme 0.17% fraud rate)
  - **ROC-AUC**: ~0.97 (Excellent ranking capability)
- **Feature Engineering**:
  - `Amount` → `Log_Amount` (to handle skewness)
  - `Time` → `Hour` (to capture daily patterns)
  - Removed low-correlation features (V13, V15, V22-V26) to prevent overfitting.

---

## 📁 Project Structure
credit-card-fraud-detection-api/
├── app.py # Main FastAPI application
├── Fraud_model.pkl # Trained Logistic Regression model
├── Scaler.pkl # Fitted StandardScaler
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── .gitignore # Files to ignore in Git
└── venv/ # Virtual environment (ignored)
step1:
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
# source venv/bin/activate

step2:
pip install -r requirements.txt
uvicorn app:app --reload
