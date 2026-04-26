import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

@st.cache_resource
def load_artifacts():
    model, scaler = None, None
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
    if os.path.exists("scaler.pkl"):
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;800&family=Lato:wght@300;400;700&display=swap');
html, body, [class*="css"] { font-family: 'Lato', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }
.stApp { background: #faf8f3; color: #2c2c2c; }
.price-box { background: linear-gradient(135deg, #1a3c5e, #2d6a9f); color: white;
             border-radius: 16px; padding: 2rem; text-align: center; margin: 1rem 0; }
.price-label { font-size: 0.9rem; letter-spacing: 3px; text-transform: uppercase; opacity: 0.8; }
.price-value { font-family: 'Playfair Display', serif; font-size: 3rem; font-weight: 800; margin: 0.5rem 0; }
div[data-testid="stSidebar"] { background: #f0ece0; }
</style>
""", unsafe_allow_html=True)

st.title("🏠 House Price Predictor")
st.markdown("*Predict house value using Linear Regression trained on your dataset.*")
st.divider()

st.sidebar.header("🏡 Property Details")

# ── Exact 7 features from your notebook (X = df.drop('House_Price', axis=1)) ──
square_footage       = st.sidebar.number_input("Square Footage",              300,  6000, 2000, step=50)
num_bedrooms         = st.sidebar.slider(      "Number of Bedrooms",          1,    5,    3)
num_bathrooms        = st.sidebar.slider(      "Number of Bathrooms",         1,    3,    2)
year_built           = st.sidebar.number_input("Year Built",                  1950, 2024, 2000, step=1)
lot_size             = st.sidebar.number_input("Lot Size (acres)",            0.5,  5.0,  2.5,  step=0.1, format="%.2f")
garage_size          = st.sidebar.slider(      "Garage Size (0–3 cars)",      0,    3,    1)
neighborhood_quality = st.sidebar.slider(      "Neighborhood Quality (1–10)", 1,    10,   5)

# ── Column ORDER matches your training X exactly ──────────────────────────────
input_data = pd.DataFrame([{
    "Square_Footage":       int(square_footage),
    "Num_Bedrooms":         int(num_bedrooms),
    "Num_Bathrooms":        int(num_bathrooms),
    "Year_Built":           int(year_built),
    "Lot_Size":             float(lot_size),
    "Garage_Size":          int(garage_size),
    "Neighborhood_Quality": int(neighborhood_quality),
}])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sqft",    f"{square_footage:,}")
c2.metric("Beds",    num_bedrooms)
c3.metric("Baths",   num_bathrooms)
c4.metric("Quality", f"{neighborhood_quality}/10")
st.divider()

if st.button("🏷️ Predict Price", use_container_width=True, type="primary"):
    if model is None:
        st.error("⚠️ model.pkl not found. Add your trained model file.")
    else:
        try:
            data = input_data.copy()
            # Your notebook used StandardScaler — apply if scaler.pkl is present
            if scaler is not None:
                data = pd.DataFrame(scaler.transform(data), columns=input_data.columns)

            raw_pred = model.predict(data)[0]

            # Your notebook applied: df['House_Price'] = np.log(df['House_Price'])
            # So we reverse with np.exp
            price = np.exp(raw_pred)

            st.markdown(f"""
            <div class="price-box">
                <div class="price-label">Estimated Property Value</div>
                <div class="price-value">${price:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.info(f"📊 Range (±10%): **${price*0.9:,.0f}** – **${price*1.1:,.0f}**")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("💡 Ensure model.pkl (and optionally scaler.pkl) are in the same folder as app.py")

with st.expander("📋 Feature vector sent to model"):
    st.dataframe(input_data, use_container_width=True)

with st.expander("ℹ️ Model Info"):
    st.markdown("""
    **Algorithm:** Linear Regression | **R²:** 0.946  
    **Preprocessing:** StandardScaler → predict → np.exp() (reverses log transform)  
    **Features (exact order):** Square_Footage · Num_Bedrooms · Num_Bathrooms · Year_Built · Lot_Size · Garage_Size · Neighborhood_Quality
    """)
