from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Customer Value Prediction API")

# Load trained artifacts
model = joblib.load('model_deployed.pkl')
encoder = joblib.load('label_encoder.pkl')

class CustomerMetrics(BaseModel):
    TotalTransactions: int
    TotalRevenue: float
    TotalQuantity: int
    AvgPrice: float
    IsReturn: int
    IsWeekend: float
    Country: str

@app.post("/predict")
def predict_value(data: CustomerMetrics):
    # Prepare input for model
    country_encoded = encoder.transform([data.Country])[0]
    
    # Derived features used in training
    avg_rev = data.TotalRevenue / data.TotalTransactions if data.TotalTransactions > 0 else 0
    avg_qty = data.TotalQuantity / data.TotalTransactions if data.TotalTransactions > 0 else 0
    
    input_data = [[
        data.TotalTransactions, data.TotalRevenue, avg_rev,
        data.TotalQuantity, avg_qty, data.AvgPrice,
        data.IsReturn, data.IsWeekend, country_encoded
    ]]
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "is_high_value": bool(prediction),
        "confidence": round(probability, 4)
    }
import uvicorn
import os
if __name__ == "__main__":
    # Render provides the PORT, we must use it.
    port = int(os.environ.get("PORT", 10000)) 
    # Must use 0.0.0.0 to be visible outside the container
    uvicorn.run(app, host="0.0.0.0", port=port)
    