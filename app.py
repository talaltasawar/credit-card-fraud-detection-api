from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

# ---------- 🔥 SUPER STYLISH SWAGGER UI CSS ----------
custom_css = """
    .swagger-ui .topbar { background: linear-gradient(135deg, #ff416c, #ff4b2b) !important; box-shadow: 0 4px 20px rgba(255, 65, 108, 0.5); }
    .swagger-ui .topbar .download-url-wrapper .select-label { color: #fff; font-weight: bold; }
    .swagger-ui .info .title { color: #ff416c; font-family: 'Segoe UI', sans-serif; text-shadow: 2px 2px 10px rgba(255, 65, 108, 0.3); }
    .swagger-ui .btn.execute { background: linear-gradient(135deg, #00b894, #00cec9) !important; border: none !important; color: white !important; font-weight: bold; border-radius: 30px !important; padding: 8px 30px !important; box-shadow: 0 4px 15px rgba(0, 206, 201, 0.4); transition: 0.3s; }
    .swagger-ui .btn.execute:hover { transform: scale(1.05); box-shadow: 0 6px 25px rgba(0, 206, 201, 0.7); }
    .swagger-ui .opblock-summary-method { background: linear-gradient(135deg, #667eea, #764ba2) !important; border-radius: 20px !important; padding: 4px 18px !important; font-weight: bold; }
    .swagger-ui .response-col_status { font-weight: bold; font-size: 1.2em; }
    .swagger-ui .model-box { border-radius: 16px; border: 2px solid #667eea; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2); }
    .swagger-ui .parameters-col_description input { border-radius: 25px !important; border: 1px solid #ddd; padding: 8px 15px !important; }
    .swagger-ui section.models { border-radius: 20px; border: 2px solid #ff416c; }
    .swagger-ui .scheme-container { background: rgba(0,0,0,0.02); border-radius: 20px; }
"""

# ---------- FASTAPI APP WITH CUSTOM CSS ----------
app = FastAPI(
    title="💳 Credit Card Fraud Detection API",
    description="🔍 Detect fraudulent transactions with Machine Learning. Built with ❤️ using FastAPI & Scikit-learn.",
    version="2.0.0 🚀",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "custom_css": custom_css  # <--- YAHAN APPLIED!
    }
)

# ---------- LOAD MODEL & SCALER ----------
try:
    Model = joblib.load('Fraud_model.pkl')
    Scaler = joblib.load('Scaler.pkl')
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    Model = None
    Scaler = None

# ---------- INPUT SCHEMA ----------
class TransactionData(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V14: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V27: float; V28: float
    Log_Amount: float
    Hour: float

# ---------- 🚀 ULTRA BEAUTIFUL HOMEPAGE (GLASS + ANIMATION) ----------
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>💳 Fraud Detection API</title>
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;900&display=swap" rel="stylesheet">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body {
                    font-family: 'Poppins', sans-serif;
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: #0f0c29;
                    overflow: hidden;
                    position: relative;
                }

                /* 🎨 Animated Background Blobs */
                .bg-blob {
                    position: absolute;
                    border-radius: 50%;
                    filter: blur(90px);
                    opacity: 0.6;
                    animation: floatBlob 15s infinite alternate ease-in-out;
                }
                .blob1 { width: 500px; height: 500px; background: #ff416c; top: -10%; left: -10%; }
                .blob2 { width: 600px; height: 600px; background: #667eea; bottom: -10%; right: -10%; animation-delay: 5s; }
                .blob3 { width: 300px; height: 300px; background: #00cec9; top: 50%; left: 50%; transform: translate(-50%, -50%); animation-delay: 10s; }

                @keyframes floatBlob {
                    0% { transform: translate(0px, 0px) scale(1); }
                    33% { transform: translate(50px, -50px) scale(1.1); }
                    66% { transform: translate(-30px, 30px) scale(0.9); }
                    100% { transform: translate(20px, -20px) scale(1.05); }
                }

                /* 🌟 Glass Card */
                .glass-card {
                    position: relative;
                    z-index: 10;
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(20px);
                    -webkit-backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.15);
                    border-radius: 40px;
                    padding: 60px 80px;
                    max-width: 700px;
                    width: 90%;
                    text-align: center;
                    box-shadow: 0 40px 80px rgba(0, 0, 0, 0.6);
                    animation: slideUp 1s ease-out;
                }

                @keyframes slideUp {
                    from { opacity: 0; transform: translateY(50px); }
                    to { opacity: 1; transform: translateY(0); }
                }

                .logo {
                    font-size: 70px;
                    margin-bottom: 10px;
                    display: block;
                    animation: pulse 2s infinite;
                }
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.1); }
                    100% { transform: scale(1); }
                }

                h1 {
                    font-size: 3.5rem;
                    font-weight: 900;
                    background: linear-gradient(135deg, #ff416c, #ff4b2b, #f7971e, #ffd200);
                    background-size: 300% 300%;
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    animation: gradShift 5s ease infinite;
                    margin-bottom: 10px;
                }
                @keyframes gradShift {
                    0% { background-position: 0% 50%; }
                    50% { background-position: 100% 50%; }
                    100% { background-position: 0% 50%; }
                }

                .subtitle {
                    color: rgba(255,255,255,0.8);
                    font-size: 1.3rem;
                    font-weight: 300;
                    letter-spacing: 1px;
                    margin-bottom: 30px;
                }

                .btn-group {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    justify-content: center;
                }

                .btn {
                    padding: 15px 40px;
                    border-radius: 60px;
                    text-decoration: none;
                    font-weight: 700;
                    font-size: 1.1rem;
                    transition: all 0.4s ease;
                    border: none;
                    cursor: pointer;
                    display: inline-flex;
                    align-items: center;
                    gap: 10px;
                    background: rgba(255,255,255,0.1);
                    color: white;
                    border: 1px solid rgba(255,255,255,0.2);
                    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
                }

                .btn-primary {
                    background: linear-gradient(135deg, #f7971e, #ffd200);
                    color: #1a1a2e;
                    border: none;
                    box-shadow: 0 8px 30px rgba(255, 210, 0, 0.3);
                }
                .btn-primary:hover {
                    transform: translateY(-5px) scale(1.02);
                    box-shadow: 0 15px 40px rgba(255, 210, 0, 0.6);
                }

                .btn-success {
                    background: linear-gradient(135deg, #00b894, #00cec9);
                    color: white;
                    border: none;
                    box-shadow: 0 8px 30px rgba(0, 206, 201, 0.3);
                }
                .btn-success:hover {
                    transform: translateY(-5px) scale(1.02);
                    box-shadow: 0 15px 40px rgba(0, 206, 201, 0.6);
                }

                .btn-secondary {
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.3);
                }
                .btn-secondary:hover {
                    background: rgba(255, 255, 255, 0.2);
                    transform: translateY(-5px);
                }

                .footer-text {
                    margin-top: 35px;
                    color: rgba(255,255,255,0.3);
                    font-size: 0.8rem;
                    letter-spacing: 1px;
                    border-top: 1px solid rgba(255,255,255,0.05);
                    padding-top: 25px;
                }
                .footer-text span { color: #ff416c; }

                /* Responsive */
                @media (max-width: 600px) {
                    .glass-card { padding: 40px 25px; }
                    h1 { font-size: 2.5rem; }
                    .btn { padding: 12px 25px; font-size: 1rem; }
                }
            </style>
        </head>
        <body>
            <!-- Floating Background Blobs -->
            <div class="bg-blob blob1"></div>
            <div class="bg-blob blob2"></div>
            <div class="bg-blob blob3"></div>

            <!-- Main Card -->
            <div class="glass-card">
                <span class="logo">💳✨</span>
                <h1>Fraud Detection</h1>
                <p class="subtitle">🔍 Real-time ML API to catch fraudsters</p>
                
                <div class="btn-group">
                    <a href="/docs" class="btn btn-primary">📖 Open Swagger UI</a>
                    <a href="/health" class="btn btn-success">❤️ Health Check</a>
                </div>

                <div class="footer-text">
                    ⚡ Built with <span>FastAPI</span> • <span>Scikit-learn</span> • <span>Joblib</span>
                </div>
            </div>
        </body>
    </html>
    """

# ---------- PREDICTION ENDPOINT ----------
@app.post("/predict")
def predict_fraud(transaction: TransactionData):
    if Model is None or Scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")

    try:
        input_data = [[
            transaction.V1, transaction.V2, transaction.V3, transaction.V4,
            transaction.V5, transaction.V6, transaction.V7, transaction.V8,
            transaction.V9, transaction.V10, transaction.V11, transaction.V12,
            transaction.V14, transaction.V16, transaction.V17, transaction.V18,
            transaction.V19, transaction.V20, transaction.V21, transaction.V27,
            transaction.V28, transaction.Log_Amount, transaction.Hour
        ]]
        scaled_data = Scaler.transform(input_data)
        prob = Model.predict_proba(scaled_data)[0][1]
        threshold = 0.85
        prediction = 1 if prob >= threshold else 0

        return {
            "fraud_probability": round(prob, 4),
            "threshold_used": threshold,
            "prediction": "🚨 FRAUD" if prediction == 1 else "✅ GENUINE",
            "risk_level": "🔴 HIGH" if prediction == 1 else "🟢 LOW",
            "emoji": "💀" if prediction == 1 else "😊",
            "message": "⚠️ This transaction looks suspicious!" if prediction == 1 else "✅ This transaction seems safe."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")

# ---------- HEALTH CHECK ----------
@app.get("/health")
def health_check():
    return {
        "status": "healthy ✅",
        "model_loaded": Model is not None,
        "scaler_loaded": Scaler is not None
    }