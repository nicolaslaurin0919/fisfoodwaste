import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LinearRegression
import datetime
from urllib.request import urlopen

# GitHub-hosted file URLs (ensure they are correct!)
repo_url = "https://raw.githubusercontent.com/nicolaslaurin0919/fisfoodwaste/main/"
purchase_file_url = repo_url + "purchase_data.csv"
consumption_file_url = repo_url + "consumption_data.csv"
food_bank_file_url = repo_url + "food_banks.csv"
fis_logo_url = repo_url + "fis_logo.jpg"

# Load FIS Logo
fis_logo = Image.open(urlopen(fis_logo_url))

# ✅ Function to Load CSV Data
@st.cache_data
def load_data():
    try:
        purchases = pd.read_csv(purchase_file_url, names=["Food", "Purchased", "Date", "Expiry"])
        purchases["Date"] = pd.to_datetime(purchases["Date"], errors="coerce")
        purchases["Expiry"] = pd.to_datetime(purchases["Expiry"], errors="coerce")
        purchases["Purchased"] = pd.to_numeric(purchases["Purchased"], errors="coerce").fillna(0)

        consumption = pd.read_csv(consumption_file_url, names=["Food", "Consumed", "Date"])
        consumption["Date"] = pd.to_datetime(consumption["Date"], errors="coerce")
        consumption["Consumed"] = pd.to_numeric(consumption["Consumed"], errors="coerce").fillna(0)

        return purchases, consumption
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ✅ Function to Load Food Banks
@st.cache_data
def load_food_banks():
    try:
        food_banks = pd.read_csv(food_bank_file_url)
        return food_banks
    except Exception as e:
        st.error(f"Error loading food bank data: {e}")
        return pd.DataFrame(columns=["Name", "Address", "Contact", "Website"])

# ✅ Sidebar with Navigation
st.sidebar.header("🚀 Explore the FIS Food AI System!")
menu_options = ["🏠 Home", "✏️ Data Entry", "📜 Inventory", "📊 Surplus", "📈 Predictive AI", "🥫 Food Banks"]

if "menu" not in st.session_state:
    st.session_state["menu"] = menu_options[0]

menu = st.sidebar.radio("📌 Choose a section:", menu_options, index=menu_options.index(st.session_state["menu"]))

# ✅ Home Page
if menu == "🏠 Home":
    st.image(fis_logo, width=300)
    st.subheader("🏠 Welcome to the FIS Food Waste AI-Management System!")

    st.write("""
    **Why is this system important for the French International School of Hong Kong (FIS)?**
    
    Every day, **significant amounts of food** are wasted in schools worldwide, including at FIS.  
    While humans can track and manage food manually, **AI can analyze patterns, predict food waste, and automate surplus allocation**.  

    **Key Benefits:**
    - **Real-time tracking**: Instantly logs purchases & consumption.
    - **Accurate predictions**: AI analyzes past trends to forecast future surplus.
    - **Automated surplus management**: AI suggests when to donate food to food banks before it expires.

    By using **AI-powered food waste tracking**, FIS can become a more sustainable school, reducing waste while helping the community!
    """)

    if st.button("➡️ Go to Data Entry"):
        st.session_state["menu"] = "✏️ Data Entry"
        st.rerun()

# ✅ Data Entry Page (Fixed)
elif menu == "✏️ Data Entry":
    st.subheader("✏️ Log Food Purchases & Consumption")

    action_type = st.radio("Select Action:", ["Purchase", "Consumption"])

    purchases, consumption = load_data()
    food_items = sorted(set(purchases["Food"].dropna().unique()) | set(consumption["Food"].dropna().unique()))
    food_items.append("➕ Add New Item...")
    selected_food = st.selectbox("Select or Add a New Food Item:", food_items)

    if selected_food == "➕ Add New Item...":
        selected_food = st.text_input("Enter New Food Item:")

    quantity = st.number_input("Enter Quantity (kg):", min_value=0.1, step=0.1)

    if action_type == "Purchase":
        purchase_date = st.date_input("Purchase Date", datetime.date.today())
        expiry_date = st.date_input("Expiry Date", datetime.date.today())

        if st.button("➕ Add Purchase"):
            new_data = pd.DataFrame([[selected_food, quantity, purchase_date, expiry_date]],
                                    columns=["Food", "Purchased", "Date", "Expiry"])
            new_data.to_csv(purchase_file_url, mode='a', index=False, header=False)
            st.success(f"✅ {quantity} kg of {selected_food} added to purchases!")

    elif action_type == "Consumption":
        consumption_date = st.date_input("Consumption Date", datetime.date.today())

        if st.button("🍽️ Log Consumption"):
            new_data = pd.DataFrame([[selected_food, quantity, consumption_date]],
                                    columns=["Food", "Consumed", "Date"])
            new_data.to_csv(consumption_file_url, mode='a', index=False, header=False)
            st.success(f"✅ {quantity} kg of {selected_food} logged as consumed!")

# ✅ Food Banks Page (Fixed)
elif menu == "🥫 Food Banks":
    st.subheader("🥫 Hong Kong Food Banks Directory")

    food_banks = load_food_banks()

    if food_banks.empty:
        st.warning("⚠️ No food banks found.")
    else:
        st.write("### 📋 Available Food Banks")
        st.dataframe(food_banks)

        # Ensure correct column names
        required_columns = {"Name", "Address", "Contact", "Website"}
        if not required_columns.issubset(food_banks.columns):
            st.error(f"⚠️ Missing required columns! Found: {food_banks.columns.tolist()}")
        else:
            for _, row in food_banks.iterrows():
                st.markdown(f"**🏛️ {row['Name']}**")
                st.write(f"📍 **Address:** {row['Address']}")
                st.write(f"📞 **Contact:** {row['Contact']}")
                st.write(f"🌐 **Website:** [{row['Website']}]({row['Website']})")

