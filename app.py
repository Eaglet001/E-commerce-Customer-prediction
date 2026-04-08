import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Customer Intelligence Pro", layout="wide")

# The exact 12 features used in the notebook training
FEATURES = [
    'TotalTransactions', 'TotalRevenue', 'AvgRevenue', 'TotalQuantity', 
    'AvgQuantity', 'AvgPrice', 'Price', 'Quantity', 'TotalAmount', 
    'IsReturn', 'IsWeekend', 'Country_Encoded'
]

# Load artifacts
@st.cache_resource
def load_assets():
    model = joblib.load('model_deployed.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return model, encoder

model, encoder = load_assets()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Processing", "Model Insights"])

# --- PAGE 1: SINGLE PREDICTION ---
if page == "Single Prediction":
    st.title("🎯 Single Customer Evaluation")
    col1, col2 = st.columns(2)
    
    with col1:
        rev = st.number_input("Total Revenue ($)", min_value=0.0, value=500.0)
        trans = st.number_input("Total Transactions", min_value=1, value=2)
        qty = st.number_input("Total Quantity", min_value=1, value=10)
    
    with col2:
        avg_p = st.number_input("Average Unit Price", min_value=0.0, value=50.0)
        country = st.selectbox("Country", encoder.classes_)
        is_weekend = st.slider("Weekend Purchase Ratio (0-1)", 0.0, 1.0, 0.5)

    if st.button("Analyze Customer"):
        # Engineering logic: mapping the 12 features
        c_enc = encoder.transform([country])[0]
        
        # We derive the 12 features the model expects
        # Note: In the notebook, TotalRevenue and TotalAmount were often used interchangeably 
        # based on the aggregation dictionary.
        input_data = pd.DataFrame([[
            trans,      # TotalTransactions
            rev,        # TotalRevenue
            rev/trans,  # AvgRevenue
            qty,        # TotalQuantity
            qty/trans,  # AvgQuantity
            avg_p,      # AvgPrice
            avg_p,      # Price (mean)
            qty,        # Quantity (sum)
            rev,        # TotalAmount (sum)
            0,          # IsReturn (sum)
            is_weekend, # IsWeekend (mean)
            c_enc       # Country_Encoded
        ]], columns=FEATURES)
        
        # .values converts the DataFrame to a raw array to match how the model was trained
        prediction = model.predict(input_data.values)[0]
        prob = model.predict_proba(input_data.values)[0][1]
        
        if prediction == 1:
            st.success(f"### High-Value Customer (Probability: {prob:.2%})")
        else:
            st.warning(f"### Standard Customer (Probability: {prob:.2%})")

# --- PAGE 2: BATCH PROCESSING ---
elif page == "Batch Processing":
    st.title("📂 Batch CSV Analysis")
    st.write(f"Upload a CSV. It MUST contain these columns: {', '.join(FEATURES[:-1])} and 'Country'")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        try:
            # Map Country to Encoded
            data['Country_Encoded'] = encoder.transform(data['Country'])
            
            # Select exactly the 12 features for prediction
            X = data[FEATURES].values
            data['Predicted_HighValue'] = model.predict(X)
            
            st.write("### Analysis Results")
            st.dataframe(data.head(10))
            
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime='text/csv')
        except Exception as e:
            st.error(f"Column Mismatch! Error: {e}")

# --- PAGE 3: MODEL INSIGHTS ---
elif page == "Model Insights":
    st.title("📊 Business Logic & Feature Importance")
    
    # Feature importance for all 12 features
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_title("What drives Customer Value?")
    plt.tight_layout()
    st.pyplot(fig)