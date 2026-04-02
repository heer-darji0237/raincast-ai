import streamlit as st
import joblib
import pandas as pd
import requests
import matplotlib.pyplot as plt
import os
from datetime import date

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RainCast AI 🌧", layout="wide")

# ---------------- LOAD ----------------
model = joblib.load("rain_model.pkl")

# FIXED CSV PATH (for deployment)
file_path = os.path.join(os.path.dirname(__file__), "weatherAUS.csv")
df = pd.read_csv(file_path)

API_KEY = "444e60e9c0c7d2c4d51287978c19eb07"

# ---------------- API FUNCTION ----------------
def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    try:
        res = requests.get(url)
        data = res.json()

        if res.status_code != 200:
            return None

        return {
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind": int(data["wind"]["speed"]),
            "cloud": int(data["clouds"]["all"] / 10),
            "rain_today": 1 if "rain" in data else 0
        }

    except:
        return None

# ---------------- HEADER ----------------
st.title("🌧 RainCast AI")
st.caption("AI Powered Rain Prediction System")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🌦 Prediction", "📊 EDA", "📂 Bulk Scanner"])

# =========================================================
# 🌦 TAB 1: SMART PREDICTION
# =========================================================
with tab1:

    st.subheader("Smart Rain Prediction")

    col1, col2 = st.columns(2)

    with col1:
        city = st.text_input("📍 Enter City", "Surat")

    with col2:
        selected_date = st.date_input("📅 Select Date")

    weather_data = get_weather(city)

    # AUTO DATA
    if selected_date == date.today() and weather_data:
        st.success("✅ Using Live Weather Data")

        temp = weather_data["temp"]
        humidity = weather_data["humidity"]
        pressure = weather_data["pressure"]
        wind = weather_data["wind"]
        cloud = weather_data["cloud"]
        rain_today_val = weather_data["rain_today"]

    else:
        st.warning("⚠️ Using Dataset Estimation")

        temp = int(df["Temp3pm"].mean())
        humidity = int(df["Humidity3pm"].mean())
        pressure = int(df["Pressure3pm"].mean())
        wind = int(df["WindSpeed3pm"].mean())
        cloud = int(df["Cloud3pm"].mean())
        rain_today_val = 0

    # SHOW VALUES
    st.info(f"""
    🌡 Temp: {temp} °C  
    💧 Humidity: {humidity}%  
    🌬 Pressure: {pressure} hPa  
    🌬 Wind: {wind} km/h  
    ☁ Cloud: {cloud}  
    🌧 Rain Today: {"Yes" if rain_today_val else "No"}
    """)

    # PREDICT
    if st.button("🚀 Predict Rain"):

        input_df = pd.DataFrame({
            "Humidity3pm":[humidity],
            "Pressure3pm":[pressure],
            "Temp3pm":[temp],
            "WindSpeed3pm":[wind],
            "RainToday":[rain_today_val],
            "Cloud3pm":[cloud]
        })

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]*100

        if pred == 1:
            st.error(f"🌧 Rain Expected ({prob:.2f}%)")
        else:
            st.success(f"☀ No Rain ({prob:.2f}%)")

# =========================================================
# 📊 TAB 2: EDA (FIXED)
# =========================================================
with tab2:

    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Temperature Distribution")
        fig, ax = plt.subplots()
        df['Temp3pm'].hist(ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("### Humidity vs Rain")
        fig, ax = plt.subplots()
        ax.scatter(df['Humidity3pm'], df['RainTomorrow'].map({"Yes":1,"No":0}))
        st.pyplot(fig)

    # ✅ FIXED FEATURE IMPORTANCE
    st.write("### Feature Importance")

    features = [
        "Humidity3pm",
        "Pressure3pm",
        "Temp3pm",
        "WindSpeed3pm",
        "RainToday",
        "Cloud3pm"
    ]

    importances = model.feature_importances_

    min_len = min(len(features), len(importances))
    features = features[:min_len]
    importances = importances[:min_len]

    fig, ax = plt.subplots()
    ax.barh(features, importances)
    st.pyplot(fig)

# =========================================================
# 📂 BULK SCANNER
# =========================================================
with tab3:

    st.subheader("Bulk Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        data = pd.read_csv(file)
        st.write("📊 Data Preview")
        st.dataframe(data)

        try:
            data["RainToday"] = data["RainToday"].map({"Yes": 1, "No": 0})

            input_data = data[[
                "Humidity3pm",
                "Pressure3pm",
                "Temp3pm",
                "WindSpeed3pm",
                "RainToday",
                "Cloud3pm"
            ]]

            input_data = input_data.fillna(input_data.mean())

            preds = model.predict(input_data)
            data["Prediction"] = preds

            st.success("✅ Prediction Done")
            st.dataframe(data)

            csv = data.to_csv(index=False).encode()
            st.download_button("⬇ Download Results", csv, "result.csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built by Heer Darji | RainCast AI 🌧")
