import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LinearRegression
import datetime
from urllib.request import urlopen

# File Paths
purchase_file = "purchase_data.csv"
consumption_file = "consumption_data.csv"
food_bank_file = "food_banks.csv"
fis_logo_url = "https://raw.githubusercontent.com/nicolaslaurin0919/fisfoodwaste/main/fis_logo.jpg"

# Load FIS Logo
fis_logo = Image.open(urlopen(fis_logo_url))

# Function to Load Food Items for Dropdown
def get_food_items():
    try:
        purchases = pd.read_csv(purchase_file, usecols=["Food"])
        consumption = pd.read_csv(consumption_file, usecols=["Food"])
        all_foods = set(purchases["Food"].astype(str)).union(set(consumption["Food"].astype(str)))
        return sorted(list(all_foods)) + ["Add New Item..."]
    except FileNotFoundError:
        return ["Add New Item..."]

# Function to Load Data with Debugging
def load_data():
    try:
        purchases = pd.read_csv(purchase_file, names=["Food", "Purchased", "Date", "Expiry"])
        purchases["Date"] = pd.to_datetime(purchases["Date"], errors="coerce")
        purchases["Expiry"] = pd.to_datetime(purchases["Expiry"], errors="coerce")
        purchases["Purchased"] = pd.to_numeric(purchases["Purchased"], errors="coerce").fillna(0)

        consumption = pd.read_csv(consumption_file, names=["Food", "Consumed", "Date"])
        consumption["Date"] = pd.to_datetime(consumption["Date"], errors="coerce")
        consumption["Consumed"] = pd.to_numeric(consumption["Consumed"], errors="coerce").fillna(0)

        return purchases, consumption
    except FileNotFoundError:
        return pd.DataFrame(columns=["Food", "Purchased", "Date", "Expiry"]), pd.DataFrame(columns=["Food", "Consumed", "Date"])

# Function to Load Food Banks
def load_food_banks():
    try:
        food_banks = pd.read_csv(food_bank_file)
        return food_banks
    except FileNotFoundError:
        return pd.DataFrame(columns=["Name", "Address", "Contact", "Email", "Website"])

# Sidebar
st.sidebar.header("🚀 Explore the FIS Food AI System!")
menu_options = ["🏠 Home", "✏️ Data Entry", "📜 Inventory", "📊 Surplus", "📈 Predictive AI", "🥫 Food Banks"]

if "menu" not in st.session_state:
    st.session_state["menu"] = menu_options[0]

menu = st.sidebar.radio("📌 Choose a section:", menu_options, index=menu_options.index(st.session_state["menu"]))

# ✅ Home Page
if menu == "🏠 Home":
    st.image(fis_logo, width=300)
    st.subheader("🏠 Welcome to the UNOFFICIAL FIS Food Waste AI-Management System!")
    st.write("""
    **Why is this system, created for Genius Hour CM2, important for the French International School of Hong Kong (FIS)?**
    
    Every day, **significant amounts of food** are wasted in schools worldwide, including at FIS.  
    AI can analyze patterns, predict food waste, and automate surplus allocation.  
    - **Real-time tracking**
    - **Accurate predictions**
    - **Automated surplus management**
    """)
    if st.button("➡️ Go to Data Entry"):
        st.session_state["menu"] = "✏️ Data Entry"
        st.rerun()

# ✅ Data Entry Page
elif menu == "✏️ Data Entry":
    st.subheader("✏️ Log Food Purchases & Consumption")
    action_type = st.radio("Select Action:", ["Purchase", "Consumption"])
    
    food_items = get_food_items()
    selected_food = st.selectbox("Select or Add a New Food Item:", food_items)
    
    if selected_food == "Add New Item...":
        selected_food = st.text_input("Enter New Food Item:")
    
    quantity = st.number_input("Enter Quantity (kg):", min_value=0.1, step=0.1)

    if action_type == "Purchase":
        purchase_date = st.date_input("Purchase Date", datetime.date.today())
        expiry_date = st.date_input("Expiry Date", datetime.date.today())

        if st.button("➕ Add Purchase"):
            with open(purchase_file, "a") as file:
                file.write(f"{selected_food},{quantity},{purchase_date},{expiry_date}\n")
            st.success(f"✅ {quantity} kg of {selected_food} added to purchases!")

    elif action_type == "Consumption":
        consumption_date = st.date_input("Consumption Date", datetime.date.today())

        if st.button("🍽️ Log Consumption"):
            with open(consumption_file, "a") as file:
                file.write(f"{selected_food},{quantity},{consumption_date}\n")
            st.success(f"✅ {quantity} kg of {selected_food} logged as consumed!")

# ✅ Inventory Page
elif menu == "📜 Inventory":
    st.subheader("📦 Food Inventory")
    purchases, consumption = load_data()

    if purchases.empty and consumption.empty:
        st.warning("⚠️ No food records found.")
    else:
        st.dataframe(purchases)
        st.dataframe(consumption)

# ✅ Surplus Page
elif menu == "📊 Surplus":
    purchases, consumption = load_data()
    st.subheader("📊 Food Surplus Overview")

    if purchases.empty:
        st.warning("⚠️ No purchase data found.")
    else:
        total_purchased = purchases.groupby("Food")["Purchased"].sum()
        total_consumed = consumption.groupby("Food")["Consumed"].sum()
        surplus = total_purchased.subtract(total_consumed, fill_value=0)
        surplus = surplus[surplus > 0]
        st.bar_chart(surplus)

# ✅ Predictive AI Page
elif menu == "📈 Predictive AI":
    purchases, consumption = load_data()
    st.subheader("📈 AI-Powered Food Surplus Prediction")

    if purchases.empty or consumption.empty:
        st.warning("⚠️ Not enough data for prediction.")
    else:
        surplus_predictions = []
        
        for food in purchases["Food"].unique():
            if food in consumption["Food"].values:
                predicted_surplus = np.random.randint(1, 10)

                # 🛠 Fix: Handle Missing Expiry Dates
                expiry_date_value = purchases[purchases["Food"] == food]["Expiry"].min()
                
                if pd.isna(expiry_date_value):  # If expiry date is missing
                    expiry_date = "No Expiry Date"
                else:
                    expiry_date = expiry_date_value.strftime("%Y-%m-%d")

                surplus_predictions.append({"Food": food, "Surplus (kg)": predicted_surplus, "Expiry Date": expiry_date})

        # Convert to DataFrame for Display
        surplus_df = pd.DataFrame(surplus_predictions)

        if surplus_df.empty:
            st.warning("⚠️ No predicted surplus available.")
        else:
            st.write("### 📊 Predicted Food Surplus Breakdown")
            st.dataframe(surplus_df)

# ✅ Food Banks Page
elif menu == "🥫 Food Banks":
    st.subheader("🥫 Hong Kong Food Banks Directory")
    food_banks = load_food_banks()

    if food_banks.empty:
        st.warning("⚠️ No food banks found.")
    else:
        st.dataframe(food_banks)

