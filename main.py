from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import shap
import motor.motor_asyncio
from datetime import datetime
import os

app = FastAPI(title="Itachi Shield API")

# السماح للواجهة الأمامية بالاتصال
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# ١. إعداد الاتصال بقاعدة بيانات MongoDB (مرن للسحابة أو المحلي)
# ---------------------------------------------------------

MONGO_DETAILS = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
db = client.fraud_db
collection = db.get_collection("transactions_history")


model = load_model('ieee_fraud_model')

@app.post("/predict")
async def predict_fraud(data: dict):
    try:
       
        df = pd.DataFrame([data])
        
        
        try:
            expected_features = model.feature_names_in_
        except:
            expected_features = model.named_steps['trained_model'].feature_names_in_

        for col in expected_features:
            if col not in df.columns and col != 'isFraud':
                df[col] = None
        
        df = df[[c for c in expected_features if c != 'isFraud']]

       
        predictions = predict_model(model, data=df)
        label = int(predictions['prediction_label'].iloc[0])
        score = float(predictions['prediction_score'].iloc[0])
        
        
        prep_data = model[:-1].transform(df)
        explainer = shap.TreeExplainer(model.named_steps['trained_model'])
        shap_values = explainer.shap_values(prep_data)
        
        if isinstance(shap_values, list): shap_values = shap_values[1]
        explanation = np.nan_to_num(shap_values[0]).tolist()







        # ---------------------------------------------------------
        # ٢. حفظ المعاملة فوراً في قاعدة بيانات MongoDB
        # ---------------------------------------------------------
        record = {
            "transaction_id": data.get("TransactionID", "Unknown"),
            "transaction_amount": data.get("TransactionAmt", 0.0),
            "timestamp": datetime.now(),
            "prediction": label,
            "confidence": score,
            "explanation": explanation
        }
        await collection.insert_one(record)

        # إرجاع النتيجة للواجهة الأمامية
        return {
            "prediction": label,
            "confidence": score,
            "explanation": explanation
        }
        
    except Exception as e:
        print(f"Server Error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # إعداد البورت ليكون مرناً مع السيرفرات السحابية أو 8000 محلياً
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)