
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Car Price Prediction")
st.caption("Enter vehicle details below, then click Predict. Categorical options come from your trained encoder when available.")

# ---------- Config: model / categories paths ----------
MODEL_PATH = Path("dataset/model_development/RFR.pkl")  # change if needed
CATEGORIES_JSON = Path("dataset/model_development/ohe_categories.json")  # optional fallback

# ---------- Load model or pipeline ----------
pipeline = None
raw_model = None
loaded = None

if MODEL_PATH.exists():
    try:
        loaded = pickle.load(open(MODEL_PATH, "rb"))
    except Exception as e:
        st.error(f"Could not load model from {MODEL_PATH}: {e}")
else:
    st.info(f"Model file not found at {MODEL_PATH}. Place your model there or update MODEL_PATH in single.py.")

# Detect sklearn Pipeline vs bare estimator
if loaded is not None:
    if hasattr(loaded, "named_steps"):
        pipeline = loaded
    else:
        raw_model = loaded

# ---------- Helper to extract categories from a Pipeline ----------
categorical_feature_names = ["seller_type", "fuel", "transmission", "brand"]

def extract_categories_from_pipeline(pipeline_obj):
    out = {}
    try:
        # Look for a ColumnTransformer in the pipeline steps
        preprocess = None
        for name, step in pipeline_obj.named_steps.items():
            if "columntransformer" in step.__class__.__name__.lower():
                preprocess = step
                break

        if preprocess is None:
            return out

        # Traverse transformers to find OneHotEncoder(s)
        def harvest_from_ct(ct, store):
            if not hasattr(ct, "transformers_"):
                return
            for trans_name, trans, cols in ct.transformers_:
                if trans is None or trans == "drop":
                    continue
                class_name = trans.__class__.__name__.lower()
                if "onehotencoder" in class_name:
                    if hasattr(trans, "categories_"):
                        for col, cats in zip(cols, trans.categories_):
                            store[col] = list(cats)
                elif hasattr(trans, "transformers_"):
                    # nested ColumnTransformer
                    harvest_from_ct(trans, store)

        harvest_from_ct(preprocess, out)
        return out
    except Exception:
        return out

# Try to read categories from pipeline if present
encoder_categories = {}
if pipeline is not None:
    encoder_categories = extract_categories_from_pipeline(pipeline)

# ---------- Fallback: JSON with categories ----------
if not encoder_categories and CATEGORIES_JSON.exists():
    try:
        encoder_categories = json.loads(CATEGORIES_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"Could not load categories from {CATEGORIES_JSON}: {e}")

# ---------- Final fallback defaults (replace with your actual training values if possible) ----------
default_categories = {
    "seller_type": ["Individual", "Dealer", "Trustmark Dealer"],
    "fuel": ["Petrol", "Diesel", "CNG", "LPG", "Electric"],
    "transmission": ["Manual", "Automatic"],
    "brand": [
        "Maruti", "Hyundai", "Honda", "Toyota", "Mahindra", "Tata",
        "Volkswagen", "Ford", "Renault", "BMW", "Audi", "Mercedes-Benz",
        "Skoda", "Nissan", "MG", "Kia"
    ],
}
# Merge—discovered categories take priority
for k, v in default_categories.items():
    encoder_categories.setdefault(k, v)

# ---------- Inputs ----------
st.subheader("Vehicle details")

col1, col2 = st.columns(2)
with col1:
    vehicle_age = st.number_input("Vehicle age (years)", min_value=0, max_value=50, value=5, step=1)
    km_driven   = st.number_input("KM driven (×1000)", min_value=0, max_value=10000, value=50, step=1)
    engine      = st.number_input("Engine size (cc)", min_value=0, max_value=8000, value=1197, step=1)
with col2:
    mileage     = st.number_input("Mileage (km/l)", min_value=0.0, max_value=100.0, value=18.0, step=0.1, format="%.1f")
    max_power   = st.number_input("Max power (bhp)", min_value=0.0, max_value=1500.0, value=82.0, step=1.0, format="%.1f")
    seats       = st.number_input("Number of seats", min_value=1, max_value=20, value=5, step=1)

st.subheader("Categorical features")
seller_type = st.selectbox("Seller type", options=encoder_categories["seller_type"])
fuel_type = st.selectbox("Fuel type", options=encoder_categories["fuel"])
transmission_type = st.selectbox("Transmission", options=encoder_categories["transmission"])
brand = st.selectbox("Brand", options=encoder_categories["brand"])

# ---------- Build model input ----------
# IMPORTANT: these column names must match training exactly
def build_input_df():
    row = {
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "seller_type": seller_type,
        "fuel": fuel_type,
        "transmission": transmission_type,
        "brand": brand,
    }
    return pd.DataFrame([row])

def manual_ohe(df_row: pd.DataFrame, categories: dict) -> np.ndarray:
    """
    Manual one-hot encoding for a single-row DataFrame when your saved model
    is a bare regressor (no Pipeline). Category order must match training!
    """
    numeric_cols = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
    cat_cols = ["seller_type", "fuel", "transmission", "brand"]

    vec = [float(df_row.iloc[0][c]) for c in numeric_cols]
    for c in cat_cols:
        cats = categories[c]
        val = df_row.iloc[0][c]
        for cat in cats:
            vec.append(1.0 if val == cat else 0.0)
    return np.array([vec], dtype=float)

# ---------- Predict ----------
if st.button("Predict price"):
    X = build_input_df()
    try:
        if pipeline is not None:
            pred = float(pipeline.predict(X)[0])
            st.success(f"Estimated price: ₹{pred:,.0f}")
        elif raw_model is not None:
            x_vec = manual_ohe(X, encoder_categories)
            pred = float(raw_model.predict(x_vec)[0])
            st.success(f"Estimated price: ₹{pred:,.0f}")
            st.caption("Used manual one-hot encoding (since a full Pipeline was not found).")
        else:
            st.error("No model loaded. Please ensure your model file exists and is a trained Pipeline or estimator.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        with st.expander("Show input row and categories (debug)"):
            st.write(X)
            st.write(encoder_categories)
else:
    st.caption("Tip: For the smoothest experience, save your preprocessing + model as a single sklearn Pipeline.")
