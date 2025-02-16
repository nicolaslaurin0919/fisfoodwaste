import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LinearRegression
import datetime

# File Paths
purchase_file = "/Users/nicolaslaurin/Documents/purchase_data.csv"
consumption_file = "/Users/nicolaslaurin/Documents/consumption_data.csv"
food_bank_file = "/Users/nicolaslaurin/Documents/food_banks.csv"
food_icon_path = "/Users/nicolaslaurin/Documents/food_icons/"
fis_logo_path = "/Users/nicolaslaurin/Documents/fis_logo.jpg"

# Load FIS Logo
fis_logo = Image.open(fis_logo_path)

# Function to Load Food Items for Dropdown
def get_food_items():
    try:
        purchases = pd.read_csv(purchase_file, usecols=["Food"])
        consumption = pd.read_csv(consumption_file, usecols=["Food"])
        all_foods = set(purchases["Food"].astype(str)).union(set(consumption["Food"].astype(str)))
        return sorted(list(all_foods)) + ["Add New Item..."]
    except FileNotFoundError:
        return ["Add New Item..."]

# Function to Load Data
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
# ✅ Function to Load Food Banks
def load_food_banks():
    try:
        food_banks = pd.read_csv(food_bank_file)
        return food_banks
    except FileNotFoundError:
        return pd.DataFrame(columns=["Name", "Address", "Contact", "Website"])

# ✅ Sidebar with Improved Visibility
st.sidebar.header("🚀 Explore the FIS Food AI System!")
menu_options = ["🏠 Home", "✏️ Data Entry", "📜 Inventory", "📊 Surplus", "📈 Predictive AI", "🥫 Food Banks"]

# Ensure session state initializes correctly
if "menu" not in st.session_state:
    st.session_state["menu"] = menu_options[0]

# Sidebar menu
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

# ✅ Inventory Page (Fixed)
elif menu == "📜 Inventory":
    st.subheader("📦 Food Inventory")

    # Load data
    purchases, consumption = load_data()

    if purchases.empty and consumption.empty:
        st.warning("⚠️ No food records found. Please add data in the 'Data Entry' section.")
    else:
        # 🛒 Display Purchase Data
        st.write("### 📋 Purchase Records")
        if not purchases.empty:
            st.dataframe(purchases)
        else:
            st.warning("❌ No purchase records found.")

        # 🍽️ Display Consumption Data
        st.write("### 🍽️ Consumption Records")
        if not consumption.empty:
            st.dataframe(consumption)
        else:
            st.warning("❌ No consumption records found.")

# ✅ Data Entry Page
elif menu == "✏️ Data Entry":
    st.subheader("✏️ Log Food Purchases & Consumption")

    # ✅ **Action Selection at the Top**
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

# ✅ Surplus Visualization
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

        if surplus.empty:
            st.warning("⚠️ No surplus available.")
        else:
            fig, ax = plt.subplots()
            surplus.plot(kind="bar", ax=ax, color="green")
            ax.set_xlabel("Food Item")
            ax.set_ylabel("Quantity (kg)")
            ax.set_title("Available Surplus by Food Type")
            st.pyplot(fig)

elif menu == "📈 Predictive AI":
    st.subheader("📈 AI-Powered Food Surplus Prediction")

    # 📜 **Explanation: How AI Predicts Surplus?**
    st.markdown("""
    ### 🤖 How Does AI Predict Food Surplus?
    The system **analyzes past food purchase & consumption trends** using **Machine Learning (Linear Regression)** to predict future surplus.  
    - **📊 Purchase Trends**: AI learns from past purchase records.
    - **🍽️ Consumption Patterns**: AI detects how fast food is used.
    - **🔮 Prediction Model**: AI estimates how much food will remain **before expiry**.

    By **continuously learning**, the system helps reduce waste and **suggests food donation** before expiry!
    """)

    # Load data
    purchases, consumption = load_data()

    if purchases.empty or consumption.empty:
        st.warning("⚠️ Not enough data for prediction.")
    else:
        # Convert dates into numerical values
        purchases["Days_Ago"] = (pd.Timestamp.today() - purchases["Date"]).dt.days
        purchases["Expiry_Days"] = (purchases["Expiry"] - pd.Timestamp.today()).dt.days
        consumption["Days_Ago"] = (pd.Timestamp.today() - consumption["Date"]).dt.days

        # Drop NaN values
        purchases = purchases.dropna(subset=["Days_Ago", "Purchased", "Expiry_Days"])
        consumption = consumption.dropna(subset=["Days_Ago", "Consumed"])

        # 📊 **Surplus Prediction Table**
        surplus_predictions = []
        
        for food in purchases["Food"].unique():
            if food in consumption["Food"].values:
                model = LinearRegression()

                # 📈 **Predict Future Purchase**
                purchase_data = purchases[purchases["Food"] == food]
                if len(purchase_data) > 1:
                    X_purchase = purchase_data["Days_Ago"].values.reshape(-1, 1)
                    y_purchase = purchase_data["Purchased"].values
                    model.fit(X_purchase, y_purchase)
                    predicted_purchase = max(0, model.predict([[0]])[0])
                else:
                    predicted_purchase = purchase_data["Purchased"].sum()

                # 📉 **Predict Future Consumption**
                consumption_data = consumption[consumption["Food"] == food]
                if len(consumption_data) > 1:
                    X_consumption = consumption_data["Days_Ago"].values.reshape(-1, 1)
                    y_consumption = consumption_data["Consumed"].values
                    model.fit(X_consumption, y_consumption)
                    predicted_consumption = max(0, model.predict([[0]])[0])
                else:
                    predicted_consumption = consumption_data["Consumed"].sum()

                # 📌 **Final Surplus Calculation**
                predicted_surplus = max(0, predicted_purchase - predicted_consumption)
                
                # 🕒 **Find Expiry Date (YYYY-MM-DD)**
                expiry_date = purchases[purchases["Food"] == food]["Expiry"].min().strftime("%Y-%m-%d")

                # Append to list
                surplus_predictions.append({"Food": food, "Surplus (kg)": round(predicted_surplus, 2), "Expiry Date": expiry_date})

        # **Convert to DataFrame for Display**
        surplus_df = pd.DataFrame(surplus_predictions)

        if surplus_df.empty:
            st.warning("⚠️ No predicted surplus available.")
        else:
            st.write("### 📊 Predicted Food Surplus Breakdown")
            st.dataframe(surplus_df)

            # 📥 **Dropdown to Offer Surplus to Food Banks**
            st.write("### 🤝 Offer Surplus to Food Banks")

            # Load food bank data
            try:
                food_banks = pd.read_csv(food_bank_file)["Name"].tolist()
            except FileNotFoundError:
                food_banks = ["Food Bank A", "Food Bank B", "Food Bank C"]

            for index, row in surplus_df.iterrows():
                food_item = row["Food"]
                surplus_qty = row["Surplus (kg)"]
                expiry = row["Expiry Date"]

                st.markdown(f"**{food_item} - {surplus_qty} kg (Expiry: {expiry})**")
                selected_bank = st.selectbox(f"🏛️ Choose a food bank for {food_item}:", ["Select a food bank"] + food_banks, key=f"food_bank_{index}")

                if st.button(f"📩 Offer {food_item} to {selected_bank}", key=f"offer_{index}"):
                    if selected_bank != "Select a food bank":
                        st.success(f"✅ {food_item} ({surplus_qty} kg) has been offered to {selected_bank}!")
                    else:
                        st.error("⚠️ Please select a valid food bank before proceeding.")
elif menu == "🥫 Food Banks":
    st.subheader("🥫 Hong Kong Food Banks Directory")

    # 📜 **Introduction: Why Donate?**
    st.markdown("""
    ### 🏛️ Why Donate to Food Banks?
    **Food banks** help redistribute surplus food to those in need, reducing waste and fighting hunger.  
    Use this directory to find **local food banks** and contact them directly to arrange food donations.
    """)

    # Load Food Bank Data
    try:
        food_banks = pd.read_csv(food_bank_file)
        if food_banks.empty:
            st.warning("⚠️ No food banks found in the database.")
        else:
            st.write("### 📋 Available Food Banks")
            st.dataframe(food_banks)

            # 📞 **Click-to-Contact**
            for index, row in food_banks.iterrows():
                bank_name = row["Name"]
                contact = row["Contact"]
                email = row["Email"]
                website = row["Website"]

                st.markdown(f"#### 🏛️ {bank_name}")
                st.write(f"📞 **Phone:** {contact}")
                st.write(f"📧 **Email:** {email}")
                st.write(f"🌍 **Website:** [{website}]({website})")

                if st.button(f"📩 Contact {bank_name}", key=f"contact_{index}"):
                    st.success(f"✅ Contact details for {bank_name} displayed above.")

    except FileNotFoundError:
        st.error("⚠️ Food bank database not found. Please ensure the file exists.")

