import streamlit as st
import joblib
import pandas as pd
import requests
import matplotlib.pyplot as plt
import os
from datetime import date

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RainCast AI 🌧", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stButton>button {
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ----------------
model = joblib.load("rain_model.pkl")

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
# 🌦 TAB 1: PREDICTION
# =========================================================
with tab1:

    st.subheader("Smart Rain Prediction")

    col1, col2 = st.columns(2)

    with col1:
        city = st.text_input("📍 Enter City", "Ahmedabad")

    with col2:
        selected_date = st.date_input("📅 Select Date")

    weather_data = get_weather(city)

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

    st.info(f"""
    🌡 Temp: {temp} °C  
    💧 Humidity: {humidity}%  
    🌬 Pressure: {pressure} hPa  
    🌬 Wind: {wind} km/h  
    ☁ Cloud: {cloud}  
    🌧 Rain Today: {"Yes" if rain_today_val else "No"}
    """)

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
# 📊 TAB 2: EDA
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

    # Feature Importance FIXED
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
# 📂 TAB 3: BULK SCANNER (PRO LEVEL)
# =========================================================
with tab3:

    st.subheader("🔍 Bulk Rain Prediction Scanner")

    # ---------------- SAMPLE DOWNLOAD ----------------
    st.markdown("### 1️⃣ Download Sample Templates")

    sample_data = pd.DataFrame({
        "Humidity3pm":[70,60],
        "Pressure3pm":[1005,1012],
        "Temp3pm":[28,32],
        "WindSpeed3pm":[15,10],
        "RainToday":[1,0],
        "Cloud3pm":[7,3]
    })

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = sample_data.to_csv(index=False).encode()
        st.download_button("📄 CSV Sample", csv, "sample.csv")

    with col2:
        excel_file = "sample.xlsx"
        sample_data.to_excel(excel_file, index=False)
        with open(excel_file, "rb") as f:
            st.download_button("📊 Excel Sample", f, "sample.xlsx")

    with col3:
        json_data = sample_data.to_json(orient="records")
        st.download_button("🧾 JSON Sample", json_data, "sample.json")

    st.markdown("---")

    # ---------------- UPLOAD ----------------
    st.markdown("### 2️⃣ Upload File to Scan")

    file = st.file_uploader(
        "Upload CSV / Excel / JSON",
        type=["csv", "xlsx", "json"]
    )

    if file:

        try:
            # File type detection
            if file.name.endswith(".csv"):
                data = pd.read_csv(file)

            elif file.name.endswith(".xlsx"):
                data = pd.read_excel(file)

            elif file.name.endswith(".json"):
                data = pd.read_json(file)

            else:
                st.error("Unsupported file type")
                st.stop()

            st.success("✅ File Uploaded")
            st.dataframe(data)

            # Preprocess
            if "RainToday" in data.columns:
                data["RainToday"] = data["RainToday"].replace({"Yes":1,"No":0})

            required_cols = [
                "Humidity3pm",
                "Pressure3pm",
                "Temp3pm",
                "WindSpeed3pm",
                "RainToday",
                "Cloud3pm"
            ]

            if not all(col in data.columns for col in required_cols):
                st.error("❌ Missing required columns")
                st.stop()

            input_data = data[required_cols]
            input_data = input_data.fillna(input_data.mean())

            # Predict
            preds = model.predict(input_data)
            probs = model.predict_proba(input_data)[:,1]

            data["Prediction"] = preds
            data["Rain Probability (%)"] = (probs * 100).round(2)

            st.success("🎯 Prediction Completed")
            st.dataframe(data)

            # Download
            result_csv = data.to_csv(index=False).encode()
            st.download_button("⬇ Download Results", result_csv, "results.csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built by Heer Darji | RainCast AI 🌧")
