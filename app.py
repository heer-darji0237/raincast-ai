import streamlit as st
import joblib
import pandas as pd
import requests
import matplotlib.pyplot as plt
import os
from datetime import date

st.set_page_config(page_title="RainCast AI 🌧", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.big-card {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    text-align: center;
    color: white;
}
.card {
    background:#f5f5f5;
    padding:20px;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ----------------
model = joblib.load("rain_model.pkl")
df = pd.read_csv("weatherAUS.csv")

API_KEY = "444e60e9c0c7d2c4d51287978c19eb07"

# ---------------- API ----------------
def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        data = requests.get(url).json()
        return {
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind": int(data["wind"]["speed"]),
            "cloud": int(data["clouds"]["all"]/10),
            "rain_today": 1 if "rain" in data else 0
        }
    except:
        return None

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio("Navigation", ["Dashboard", "Prediction", "EDA", "Bulk Scanner", "About"])

# =========================================================
# 📊 DASHBOARD
# =========================================================
if menu == "Dashboard":

    st.markdown('<div class="big-card"><h1>🌧 RainCast AI</h1><p>AI-Based Rain Prediction</p></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))
    c2.metric("Avg Temp", f"{int(df['Temp3pm'].mean())}°C")
    c3.metric("Avg Humidity", f"{int(df['Humidity3pm'].mean())}%")
    c4.metric("Avg Pressure", f"{int(df['Pressure3pm'].mean())}")

# =========================================================
# 🌦 PREDICTION
# =========================================================
elif menu == "Prediction":

    st.markdown('<div class="big-card"><h1>🌧 Smart Rain Prediction</h1></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        city = st.text_input("📍 Enter City", "Ahmedabad")

    with col2:
        selected_date = st.date_input("📅 Select Date")

    # -------- FETCH WEATHER AUTOMATICALLY --------
    weather = None
    if city:
        weather = get_weather(city.strip())

    # -------- DEFAULT VALUES --------
    if weather:
        temp_default = int(weather["temp"])
        humidity_default = int(weather["humidity"])
        pressure_default = int(weather["pressure"])
        wind_default = int(weather["wind"])
        cloud_default = int(weather["cloud"])
        rain_today_default = int(weather["rain_today"])

        st.success("✅ Live weather data loaded")

    else:
        temp_default = int(df["Temp3pm"].mean())
        humidity_default = int(df["Humidity3pm"].mean())
        pressure_default = int(df["Pressure3pm"].mean())
        wind_default = int(df["WindSpeed3pm"].mean())
        cloud_default = int(df["Cloud3pm"].mean())
        rain_today_default = 0

        if city != "":
            st.error("❌ City not found or API issue")

    # -------- MANUAL EDIT --------
    st.markdown("### ⚙ Adjust Values (Optional)")

    col1, col2, col3 = st.columns(3)

    with col1:
        temp = st.slider("🌡 Temperature", 0, 60, temp_default)
        humidity = st.slider("💧 Humidity", 0, 100, humidity_default)

    with col2:
        pressure = st.slider("🌬 Pressure", 900, 1100, pressure_default)
        wind = st.slider("💨 Wind Speed", 0, 150, wind_default)

    with col3:
        cloud = st.slider("☁ Cloud", 0, 10, cloud_default)
        rain_today = st.selectbox("🌧 Rain Today", [0,1], index=rain_today_default)

    # -------- DISPLAY --------
    st.info(f"""
    🌡 Temp: {temp} °C  
    💧 Humidity: {humidity}%  
    🌬 Pressure: {pressure} hPa  
    💨 Wind: {wind} km/h  
    ☁ Cloud: {cloud}  
    🌧 Rain Today: {"Yes" if rain_today else "No"}
    """)

    # -------- PREDICT --------
    if st.button("🚀 Predict Rain"):

        input_df = pd.DataFrame({
            "Humidity3pm":[humidity],
            "Pressure3pm":[pressure],
            "Temp3pm":[temp],
            "WindSpeed3pm":[wind],
            "RainToday":[rain_today],
            "Cloud3pm":[cloud]
        })

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]*100

        if pred == 1:
            st.error(f"🌧 Rain Expected ({prob:.2f}%)")
        else:
            st.success(f"☀ No Rain ({prob:.2f}%)")

# =========================================================
# 📊 EDA (NOW 4 GRAPHS)
# =========================================================
elif menu == "EDA":

    st.markdown('<div class="big-card"><h1>📊 Exploratory Data Analysis</h1></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # 1️⃣ Temp Distribution
    with col1:
        st.subheader("🌡 Temp Distribution")
        fig, ax = plt.subplots()
        df["Temp3pm"].hist(ax=ax)
        st.pyplot(fig)

    # 2️⃣ Humidity vs Rain
    with col2:
        st.subheader("💧 Humidity vs Rain")
        fig, ax = plt.subplots()
        ax.scatter(df["Humidity3pm"], df["RainTomorrow"].map({"Yes":1,"No":0}))
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    # 3️⃣ Pressure vs Temp
    with col3:
        st.subheader("🌬 Pressure vs Temp")
        fig, ax = plt.subplots()
        ax.scatter(df["Pressure3pm"], df["Temp3pm"])
        st.pyplot(fig)

    # 4️⃣ Cloud vs Rain
    with col4:
        st.subheader("☁ Cloud vs Rain")
        fig, ax = plt.subplots()
        ax.scatter(df["Cloud3pm"], df["RainTomorrow"].map({"Yes":1,"No":0}))
        st.pyplot(fig)

# =========================================================
# 📂 BULK SCANNER
# =========================================================
elif menu == "Bulk Scanner":

    st.markdown('<div class="big-card"><h1>📂 Bulk Scanner</h1></div>', unsafe_allow_html=True)

    sample = pd.DataFrame({
        "Humidity3pm":[70],
        "Pressure3pm":[1005],
        "Temp3pm":[28],
        "WindSpeed3pm":[15],
        "RainToday":[1],
        "Cloud3pm":[7]
    })

    st.subheader("1. Download Samples")

    c1, c2, c3 = st.columns(3)
    c1.download_button("CSV", sample.to_csv(index=False), "sample.csv")

    sample.to_excel("sample.xlsx", index=False)
    with open("sample.xlsx","rb") as f:
        c2.download_button("Excel", f, "sample.xlsx")

    c3.download_button("JSON", sample.to_json(orient="records"), "sample.json")

    st.subheader("2. Upload File")

    file = st.file_uploader("Upload", type=["csv","xlsx","json"])

    if file:
        if file.name.endswith(".csv"):
            data = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            data = pd.read_excel(file)
        else:
            data = pd.read_json(file)

        preds = model.predict(data)
        data["Prediction"] = preds
        st.dataframe(data)

# =========================================================
# 📘 ABOUT (MOBILESPHERE STYLE)
# =========================================================
else:

    st.markdown("""
    <div class="big-card">
        <h1>📘 About RainCast AI</h1>
        <p>AI-Based Rain Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    RainCast AI is an AI-based system developed to predict rainfall using machine learning techniques.
    It analyzes Temperature, Humidity, Pressure, Wind Speed, and Cloud Cover.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        <h3>🎯 Key Features</h3>
        <ul>
        <li>Accurate rain prediction</li>
        <li>Auto + Manual input</li>
        <li>Interactive dashboard</li>
        <li>Bulk prediction</li>
        <li>Modern UI</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h3>🛠 Technologies</h3>
        <ul>
        <li>Python</li>
        <li>Pandas & NumPy</li>
        <li>Machine Learning</li>
        <li>Streamlit</li>
        <li>Matplotlib</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
