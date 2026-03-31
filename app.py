import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RainCast AI 🌧", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("rain_model.pkl")

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c92d2);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    color: white;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.glass {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 30px;
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}

.title {
    text-align: center;
    font-size: 55px;
    font-weight: 800;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    opacity: 0.7;
    margin-bottom: 30px;
}

.stButton>button {
    background: linear-gradient(90deg, #00f2fe, #4facfe);
    color: white;
    padding: 12px 30px;
    font-size: 18px;
    border-radius: 30px;
    border: none;
}

.result {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">RainCast AI 🌧</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered Rain Prediction System</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
col1, col2, col3 = st.columns(3)

with col1:
    humidity = st.slider("Humidity (%)", 0, 100, 60)

with col2:
    pressure = st.number_input("Pressure (hPa)", 950.0, 1050.0, 1000.0)

with col3:
    temp = st.slider("Temperature (°C)", -10, 50, 25)

predict = st.button("🚀 Predict Weather")

# ---------------- PREDICTION ----------------
if predict:

    # EXACT SAME FEATURES AS TRAINING
    input_data = pd.DataFrame({
        "Humidity3pm": [humidity],
        "Pressure3pm": [pressure],
        "Temp3pm": [temp]
    })

    prediction = model.predict(input_data)[0]

    # Probability (safe)
    try:
        probability = model.predict_proba(input_data)[0][1] * 100
    except:
        probability = 0

    # 🧠 IMPROVED LOGIC (fix always sunny issue)
    if probability > 50:
        prediction = 1
    else:
        prediction = 0

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(
            f'<div class="glass result" style="background: rgba(255,0,0,0.3);">🌧 It WILL Rain Tomorrow</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="glass result" style="background: rgba(0,255,150,0.3);">☀ Clear & Sunny Tomorrow</div>',
            unsafe_allow_html=True
        )

    # Probability Bar
    st.markdown(f"""
    <div class="glass">
        <h3 style="text-align:center;">Rain Probability</h3>
        <div style="background:rgba(255,255,255,0.1); border-radius:20px;">
            <div style="width:{probability}%; background:#00f2fe; padding:10px; border-radius:20px;">
                {round(probability,2)}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; opacity:0.5;'>Built by Heer Darji | RainCast AI</div>",
    unsafe_allow_html=True
)