import streamlit as st
import joblib
import pandas as pd
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RainCast AI 🌧", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("rain_model.pkl")

# ---------------- API CONFIG ----------------
API_KEY = "444e60e9c0c7d2c4d51287978c19eb07"

# ---------------- FUNCTION ----------------
def get_weather_data(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            return None, None, None

        humidity = data['main']['humidity']
        pressure = data['main']['pressure']
        temp = data['main']['temp']

        return humidity, pressure, temp

    except:
        return None, None, None

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

# ---------------- INPUTS ----------------
selected_date = st.date_input("📅 Select Date")
city = st.text_input("📍 Enter City", "Surat")

# ---------------- FETCH WEATHER ----------------
humidity, pressure, temp = get_weather_data(city)

# ---------------- DISPLAY DATA ----------------
if humidity is not None:
    st.markdown(f"""
    <div class="glass">
        <h3 style="text-align:center;">📊 Weather Data for {city}</h3>
        <p style="text-align:center;">
        🌡 Temperature: {temp} °C<br>
        💧 Humidity: {humidity}%<br>
        🌬 Pressure: {pressure} hPa
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("❌ Failed to fetch weather data. Check API key, city name, or internet.")

# ---------------- BUTTON ----------------
predict = st.button("🚀 Predict Weather")

# ---------------- PREDICTION ----------------
if predict and humidity is not None:

    input_data = pd.DataFrame({
        "Humidity3pm": [humidity],
        "Pressure3pm": [pressure],
        "Temp3pm": [temp]
    })

    prediction = model.predict(input_data)[0]

    try:
        probability = model.predict_proba(input_data)[0][1] * 100
    except:
        probability = 0

    st.markdown("<br>", unsafe_allow_html=True)

    # RESULT
    if prediction == 1:
        st.markdown(
            '<div class="glass result" style="background: rgba(255,0,0,0.3);">🌧 It WILL Rain Tomorrow</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="glass result" style="background: rgba(0,255,150,0.3);">☀ Clear & Sunny Tomorrow</div>',
            unsafe_allow_html=True
        )

    # PROBABILITY BAR
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