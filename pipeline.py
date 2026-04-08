import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def feature_engineering(df):
    """Replicates the notebook's feature engineering steps."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['TotalAmount'] = df['Price'] * df['Quantity']
    df['IsReturn'] = df['Quantity'] < 0
    df['IsWeekend'] = df['Date'].dt.dayofweek >= 5
    
    # Customer level aggregations
    customer_data = df.groupby('CustomerNo').agg({
        'Price': 'mean',
        'Quantity': 'sum',
        'TotalAmount': 'sum',
        'IsReturn': 'sum',
        'IsWeekend': 'mean',
        'Country': 'first'
    }).reset_index()
    
    # Additional Metrics from notebook
    customer_data['TotalTransactions'] = df.groupby('CustomerNo')['TransactionNo'].nunique().values
    customer_data['TotalRevenue'] = customer_data['TotalAmount']
    customer_data['AvgRevenue'] = customer_data['TotalRevenue'] / customer_data['TotalTransactions']
    customer_data['TotalQuantity'] = customer_data['Quantity']
    customer_data['AvgQuantity'] = customer_data['TotalQuantity'] / customer_data['TotalTransactions']
    customer_data['AvgPrice'] = customer_data['Price']
    
    return customer_data

def train_production_model(df):
    """Trains the Random Forest model for High-Value Customer prediction."""
    customer_df = feature_engineering(df)
    
    # Define Target: Top 25% by Revenue
    threshold = customer_df['TotalRevenue'].quantile(0.75)
    customer_df['HighValueCustomer'] = (customer_df['TotalRevenue'] > threshold).astype(int)
    
    # Encode Country
    le = LabelEncoder()
    customer_df['Country_Encoded'] = le.fit_transform(customer_df['Country'])
    
    # Feature Selection
    features = ['TotalTransactions', 'TotalRevenue', 'AvgRevenue', 'TotalQuantity', 
                'AvgQuantity', 'AvgPrice', 'IsReturn', 'IsWeekend', 'Country_Encoded']
    
    X = customer_df[features].fillna(0)
    y = customer_df['HighValueCustomer']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save Artifacts
    joblib.dump(model, 'model_deployed.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    return model