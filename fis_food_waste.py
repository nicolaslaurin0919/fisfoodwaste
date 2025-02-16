import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from urllib.request import urlopen
from sklearn.linear_model import LinearRegression
import datetime

# 📌 GITHUB FILE PATHS (Ensure files are in your repository)
GITHUB_REPO = "https://raw.githubusercontent.com/nicolaslaurin0919/fisfoodwaste/main/"
fis_logo_url = GITHUB_REPO + "fis_logo.jpg"
purchase_file_url = GITHUB_REPO + "purchase_data.csv"
consumption_file_url = GITHUB_REPO + "consumption_data.csv"
food_bank_file_url = GITHUB_REPO + "food_banks.csv"

# ✅ Load FIS Logo
fis_logo = Image.open(urlopen(fis_logo_url))

# ✅ Function to Load Data from GitHub CSVs
@st.cache_data
def load_data():
    try:
        purchases = pd.read_csv(purchase_file_url)
        purchases["Date"] = pd.to_datetime(purchases["Date"], errors="coerce")
        purchases["Expiry"] = pd.to_datetime(purchases["Expiry"], errors="coerce")
        purchases["Purchased"] = pd.to_numeric(purchases["Purchased"], errors="coerce").fillna(0)

        consumption = pd.read_csv(consumption_file_url)
        consumption["Date"] = pd.to_datetime(consumption["Date"], errors="coerce")
        consumption["Consumed"] = pd.to_numeric(consumption["Consumed"], errors="coerce").fillna(0)

        return purchases, consumption
    except:
        return pd.DataFrame(), pd.DataFrame()

# ✅ Function to Load Food Bank Data
@st.cache_data
def load_food_banks():
    try:
        return pd.read_csv(food_bank_file_url)
    except:
        return pd.DataFrame(columns=["Name", "Address", "Contact", "Website"])

# ✅ Sidebar Navigation
st.sidebar.header("🚀 Explore the FIS Food AI System!")
menu_options = ["🏠 Home", "✏️ Data Entry", "📜 Inventory", "📊 Surplus", "📈 Predictive AI", "🥫 Food Banks"]
menu = st.sidebar.radio("📌 Choose a section:", menu_options)

# ✅ Home Page
if menu == "🏠 Home":
    st.image(fis_logo, width=300)
    st.subheader("🏠 Welcome to the FIS Food Waste AI-Management System!")

    st.write("""
    **Why is this system important for the French International School of Hong Kong (FIS)?**
    
    Every day, **significant amounts of food** are wasted. AI can **analyze patterns, predict food waste, and automate surplus allocation**.  

    **Key Benefits:**
    - **Real-time tracking**: Instantly logs purchases & consumption.
    - **Accurate predictions**: AI analyzes past trends to forecast future surplus.
    - **Automated surplus management**: AI suggests when to donate food to food banks before it expires.

    By using **AI-powered food waste tracking**, FIS can become a more sustainable school, reducing waste while helping the community!
    """)

    if st.button("➡️ Go to Data Entry"):
        st.session_state["menu"] = "✏️ Data Entry"
        st.rerun()

# ✅ Inventory Page
elif menu == "📜 Inventory":
    st.subheader("📦 Food Inventory")
    purchases, consumption = load_data()

    if purchases.empty and consumption.empty:
        st.warning("⚠️ No food records found. Please add data in 'Data Entry'.")
    else:
        st.write("### 📋 Purchase Records")
        if not purchases.empty:
            st.dataframe(purchases)

        st.write("### 🍽️ Consumption Records")
        if not consumption.empty:
            st.dataframe(consumption)

# ✅ Surplus Page
elif menu == "📊 Surplus":
    st.subheader("📊 Food Surplus Overview")
    purchases, consumption = load_data()

    if purchases.empty:
        st.warning("⚠️ No purchase data found.")
    else:
        total_purchased = purchases.groupby("Food")["Purchased"].sum()
        total_consumed = consumption.groupby("Food")["Consumed"].sum()
        surplus = total_purchased.subtract(total_consumed, fill_value=0)
        surplus = surplus[surplus > 0]

        if surplus.empty:
            st.warning("⚠️ No surplus available.")
        else:
            st.bar_chart(surplus)

# ✅ Predictive AI Page
elif menu == "📈 Predictive AI":
    st.subheader("📈 AI-Powered Food Surplus Prediction")

    st.markdown("""
    ### 🤖 How Does AI Predict Food Surplus?
    The system **analyzes past food purchase & consumption trends** using **Machine Learning (Linear Regression)** to predict future surplus.
    """)

    purchases, consumption = load_data()

    if purchases.empty or consumption.empty:
        st.warning("⚠️ Not enough data for prediction.")
    else:
        purchases["Days_Ago"] = (pd.Timestamp.today() - purchases["Date"]).dt.days
        purchases["Expiry_Days"] = (purchases["Expiry"] - pd.Timestamp.today()).dt.days
        consumption["Days_Ago"] = (pd.Timestamp.today() - consumption["Date"]).dt.days

        purchases = purchases.dropna(subset=["Days_Ago", "Purchased", "Expiry_Days"])
        consumption = consumption.dropna(subset=["Days_Ago", "Consumed"])

        surplus_predictions = []

        for food in purchases["Food"].unique():
            if food in consumption["Food"].values:
                model = LinearRegression()

                # Predict Purchases
                purchase_data = purchases[purchases["Food"] == food]
                if len(purchase_data) > 1:
                    model.fit(purchase_data[["Days_Ago"]], purchase_data["Purchased"])
                    predicted_purchase = max(0, model.predict([[0]])[0])
                else:
                    predicted_purchase = purchase_data["Purchased"].sum()

                # Predict Consumption
                consumption_data = consumption[consumption["Food"] == food]
                if len(consumption_data) > 1:
                    model.fit(consumption_data[["Days_Ago"]], consumption_data["Consumed"])
                    predicted_consumption = max(0, model.predict([[0]])[0])
                else:
                    predicted_consumption = consumption_data["Consumed"].sum()

                # Calculate Surplus
                predicted_surplus = max(0, predicted_purchase - predicted_consumption)
                expiry_date = purchases[purchases["Food"] == food]["Expiry"].min().strftime("%Y-%m-%d")
                surplus_predictions.append({"Food": food, "Surplus (kg)": round(predicted_surplus, 2), "Expiry Date": expiry_date})

        surplus_df = pd.DataFrame(surplus_predictions)

        if not surplus_df.empty:
            st.dataframe(surplus_df)

# ✅ Food Banks Page
elif menu == "🥫 Food Banks":
    st.subheader("🥫 Hong Kong Food Banks Directory")
    food_banks = load_food_banks()

    if food_banks.empty:
        st.warning("⚠️ No food banks found.")
    else:
        st.dataframe(food_banks)

        for _, row in food_banks.iterrows():
            st.markdown(f"**🏛️ {row['Name']}**")
            st.write(f"📍 **Address:** {row['Address']}")
            st.write(f"📞 **Contact:** {row['Contact']}")
            st.write(f"🌐 **Website:** [{row['Website']}]({row['Website']})")

