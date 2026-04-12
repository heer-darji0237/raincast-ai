import streamlit as st
import joblib
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
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

/* Metric card style */
.metric-card {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.metric-card h2 {
    font-size: 2rem;
    margin: 0;
    color: #7dd3fc;
}
.metric-card p {
    margin: 4px 0 0;
    font-size: 0.9rem;
    opacity: 0.85;
}

/* About section */
.about-card {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 14px;
    padding: 28px;
    color: white;
    margin-bottom: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.about-card h3 {
    color: #7dd3fc;
    margin-bottom: 8px;
}
.tech-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #7dd3fc;
    border-radius: 20px;
    padding: 4px 14px;
    margin: 4px;
    font-size: 0.85rem;
    border: 1px solid #2d6a9f;
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
tab4, tab1, tab2, tab3, tab5 = st.tabs([
    "📈 Dashboard",
    "🌦 Prediction",
    "📊 EDA",
    "📂 Bulk Scanner",
    "ℹ️ About"
])

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
    """)

    if st.button("🚀 Predict Rain"):
        input_df = pd.DataFrame({
            "Humidity3pm": [humidity],
            "Pressure3pm": [pressure],
            "Temp3pm": [temp],
            "WindSpeed3pm": [wind],
            "RainToday": [0],   # required by model, hidden from UI
            "Cloud3pm": [cloud]
        })

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] * 100

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
        ax.scatter(df['Humidity3pm'], df['RainTomorrow'].map({"Yes": 1, "No": 0}))
        st.pyplot(fig)

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
# 📂 TAB 3: BULK SCANNER
# =========================================================
with tab3:

    st.subheader("🔍 Bulk Rain Prediction Scanner")

    st.markdown("### 1️⃣ Download Sample Templates")

    sample_data = pd.DataFrame({
        "Humidity3pm": [70, 60],
        "Pressure3pm": [1005, 1012],
        "Temp3pm": [28, 32],
        "WindSpeed3pm": [15, 10],
        "RainToday": [1, 0],
        "Cloud3pm": [7, 3]
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

    st.markdown("### 2️⃣ Upload File to Scan")

    file = st.file_uploader(
        "Upload CSV / Excel / JSON",
        type=["csv", "xlsx", "json"]
    )

    if file:
        try:
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

            if "RainToday" in data.columns:
                data["RainToday"] = data["RainToday"].replace({"Yes": 1, "No": 0})

            required_cols = [
                "Humidity3pm", "Pressure3pm", "Temp3pm",
                "WindSpeed3pm", "RainToday", "Cloud3pm"
            ]

            if not all(col in data.columns for col in required_cols):
                st.error("❌ Missing required columns")
                st.stop()

            input_data = data[required_cols]
            input_data = input_data.fillna(input_data.mean())

            preds = model.predict(input_data)
            probs = model.predict_proba(input_data)[:, 1]

            data["Prediction"] = preds
            data["Rain Probability (%)"] = (probs * 100).round(2)

            st.success("🎯 Prediction Completed")
            st.dataframe(data)

            result_csv = data.to_csv(index=False).encode()
            st.download_button("⬇ Download Results", result_csv, "results.csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# =========================================================
# 📈 TAB 4: DASHBOARD
# =========================================================
with tab4:

    st.subheader("📈 Dataset Dashboard & Insights")

    # ---- Prep clean data ----
    df_clean = df.copy()
    df_clean["RainTomorrow_bin"] = df_clean["RainTomorrow"].map({"Yes": 1, "No": 0})
    df_clean["RainToday_bin"] = df_clean["RainToday"].map({"Yes": 1, "No": 0})
    df_clean = df_clean.dropna(subset=["RainTomorrow_bin"])

    total_records = len(df_clean)
    rain_days = int(df_clean["RainTomorrow_bin"].sum())
    no_rain_days = total_records - rain_days
    rain_pct = round(rain_days / total_records * 100, 1)
    avg_humidity = round(df_clean["Humidity3pm"].mean(), 1)
    avg_temp = round(df_clean["Temp3pm"].mean(), 1)
    avg_wind = round(df_clean["WindSpeed3pm"].mean(), 1)

    # ---- KPI Row ----
    st.markdown("### 🔢 Key Statistics")
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    k1.metric("📦 Total Records", f"{total_records:,}")
    k2.metric("🌧 Rainy Days", f"{rain_days:,}")
    k3.metric("☀️ Dry Days", f"{no_rain_days:,}")
    k4.metric("🌧 Rain Rate", f"{rain_pct}%")
    k5.metric("💧 Avg Humidity", f"{avg_humidity}%")
    k6.metric("🌡 Avg Temp", f"{avg_temp} °C")

    st.markdown("---")

    # ---- Row 1: Rain distribution + Monthly trend ----
    st.markdown("### 📊 Rain Distribution & Trends")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Rain Tomorrow – Class Distribution**")
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ["No Rain ☀️", "Rain 🌧"]
        sizes = [no_rain_days, rain_days]
        colors = ["#f0a500", "#2d6a9f"]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%",
            colors=colors, startangle=140,
            textprops={"fontsize": 11}
        )
        ax.set_title("Rain Tomorrow Distribution")
        st.pyplot(fig)

    with col2:
        st.write("**Average Humidity by Cloud Cover Level**")
        cloud_humidity = df_clean.groupby("Cloud3pm")["Humidity3pm"].mean().dropna()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(cloud_humidity.index.astype(str), cloud_humidity.values, color="#2d6a9f")
        ax.set_xlabel("Cloud Cover (0–9)")
        ax.set_ylabel("Avg Humidity (%)")
        ax.set_title("Cloud Cover vs Avg Humidity")
        st.pyplot(fig)

    st.markdown("---")

    # ---- Row 2: Temp vs Humidity + Wind distribution ----
    st.markdown("### 🌡 Temperature, Humidity & Wind Analysis")
    col3, col4 = st.columns(2)

    with col3:
        st.write("**Temp 3pm Distribution – Rain vs No Rain**")
        fig, ax = plt.subplots(figsize=(5, 4))
        rain_group = df_clean[df_clean["RainTomorrow_bin"] == 1]["Temp3pm"].dropna()
        no_rain_group = df_clean[df_clean["RainTomorrow_bin"] == 0]["Temp3pm"].dropna()
        ax.hist(no_rain_group, bins=30, alpha=0.6, label="No Rain", color="#f0a500")
        ax.hist(rain_group, bins=30, alpha=0.6, label="Rain", color="#2d6a9f")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.set_title("Temperature Distribution by Rain Outcome")
        st.pyplot(fig)

    with col4:
        st.write("**Wind Speed Distribution**")
        fig, ax = plt.subplots(figsize=(5, 4))
        df_clean["WindSpeed3pm"].dropna().hist(bins=30, ax=ax, color="#2d6a9f", edgecolor="white")
        ax.set_xlabel("Wind Speed (km/h)")
        ax.set_ylabel("Frequency")
        ax.set_title("Wind Speed 3pm Distribution")
        st.pyplot(fig)

    st.markdown("---")

    # ---- Row 3: Correlation heatmap ----
    st.markdown("### 🔗 Feature Correlation Heatmap")

    num_cols = ["Humidity3pm", "Pressure3pm", "Temp3pm", "WindSpeed3pm", "Cloud3pm", "RainTomorrow_bin"]
    corr = df_clean[num_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    labels_short = ["Humidity", "Pressure", "Temp", "Wind", "Cloud", "Rain"]
    ax.set_xticklabels(labels_short, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels_short, fontsize=9)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im, ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

    st.markdown("---")

    # ---- Row 4: Top rainy locations ----
    st.markdown("### 📍 Top Rainy Locations in Dataset")
    if "Location" in df_clean.columns:
        location_rain = (
            df_clean.groupby("Location")["RainTomorrow_bin"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        location_rain.columns = ["Location", "Rain Rate"]
        location_rain["Rain Rate (%)"] = (location_rain["Rain Rate"] * 100).round(1)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(location_rain["Location"][::-1], location_rain["Rain Rate (%)"][::-1], color="#2d6a9f")
        ax.set_xlabel("Rain Rate (%)")
        ax.set_title("Top 10 Locations by Rain Rate")
        st.pyplot(fig)
    else:
        st.info("Location column not available in dataset.")

# =========================================================
# ℹ️ TAB 5: ABOUT
# =========================================================
with tab5:

    st.subheader("ℹ️ About RainCast AI")

    # Hero section
    st.markdown("""
    <div class="about-card">
        <h3>🌧 RainCast AI</h3>
        <p style="font-size:1.05rem; line-height:1.7;">
            RainCast AI is an intelligent weather prediction system that uses machine learning
            to forecast rain for the next day. It combines live weather data from OpenWeatherMap API
            with a trained Random Forest model built on the <b>Australian Weather Dataset (weatherAUS)</b>
            to deliver accurate, real-time predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="about-card">
            <h3>⚙️ How It Works</h3>
            <ol style="line-height: 2;">
                <li>User enters a city & date</li>
                <li>Live weather is fetched via OpenWeatherMap API</li>
                <li>Data is passed to the trained ML model</li>
                <li>Model predicts rain probability for tomorrow</li>
                <li>Result shown with confidence percentage</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>📦 Dataset Info</h3>
            <ul style="line-height: 2;">
                <li><b>Source:</b> Australian Bureau of Meteorology</li>
                <li><b>File:</b> weatherAUS.csv</li>
                <li><b>Records:</b> ~145,000 rows</li>
                <li><b>Target:</b> RainTomorrow (Yes / No)</li>
                <li><b>Features used:</b> Humidity, Pressure, Temp, Wind, Cloud, RainToday</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card">
            <h3>🤖 Model Details</h3>
            <ul style="line-height: 2;">
                <li><b>Algorithm:</b> Random Forest Classifier</li>
                <li><b>Library:</b> scikit-learn</li>
                <li><b>Saved as:</b> rain_model.pkl (joblib)</li>
                <li><b>Input Features:</b> 6 weather parameters</li>
                <li><b>Output:</b> Rain / No Rain + probability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>🛠️ Tech Stack</h3>
            <div style="margin-top:10px;">
                <span class="tech-badge">🐍 Python</span>
                <span class="tech-badge">🎈 Streamlit</span>
                <span class="tech-badge">🤖 scikit-learn</span>
                <span class="tech-badge">🐼 Pandas</span>
                <span class="tech-badge">📊 Matplotlib</span>
                <span class="tech-badge">🌐 OpenWeatherMap API</span>
                <span class="tech-badge">📦 Joblib</span>
                <span class="tech-badge">🔢 NumPy</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Features list
    st.markdown("""
    <div class="about-card">
        <h3>✨ App Features</h3>
        <div style="display: flex; gap: 40px; flex-wrap: wrap; line-height: 2.2; font-size: 0.97rem;">
            <div>
                ✅ Live weather data integration<br>
                ✅ Single city rain prediction<br>
                ✅ Bulk CSV / Excel / JSON prediction<br>
            </div>
            <div>
                ✅ Exploratory Data Analysis (EDA)<br>
                ✅ Interactive Dashboard with KPIs<br>
                ✅ Downloadable result files<br>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Creator
    st.markdown("""
    <div class="about-card" style="text-align:center;">
        <h3>👩‍💻 Built By</h3>
        <p style="font-size: 1.3rem; font-weight: bold; color: #7dd3fc;">Heer Darji</p>
        <p style="opacity: 0.8;">RainCast AI — Bringing AI to Weather Forecasting 🌧</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built by Heer Darji | RainCast AI 🌧")
