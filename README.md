# E-Commerce High-Value Customer Prediction

## Overview
This production pipeline utilizes a **Random Forest Classifier** to identify "High-Value Customers" based on transactional behavior. 
Based on the April 2026 data analysis, a customer is considered "High Value" if their total revenue is in the top 25th percentile.

## Model Performance
- **Primary Algorithm**: Random Forest
- **Key Features**: Total Revenue, Average Order Value, Return Frequency, and Geographic Location.
- **Target Variable**: `HighValueCustomer` (Binary)

## How to Run
1. **Setup**: `pip install -r requirements.txt`
2. **Train**: Run the Jupyter Notebook to generate `.pkl` files.
3. **Start API**: `uvicorn main:app --reload`
4. **Start UI**: `streamlit run app.py`