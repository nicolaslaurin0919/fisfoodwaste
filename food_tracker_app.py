import streamlit as st
import pandas as pd
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# File names
purchase_file = "purchase_data.csv"
consumption_file = "consumption_data.csv"

# Function to add food purchase
def add_purchase():
    st.subheader("🛒 Add Food Purchase")
    food_name = st.text_input("Enter food name:")
    quantity = st.number_input("Enter weight in kg:", min_value=0.1, step=0.1)
    purchase_date = st.date_input("Enter purchase date:", datetime.date.today())

    if st.button("Add Purchase"):
        with open(purchase_file, "a") as file:
            file.write(f"{food_name},{quantity},{purchase_date}\n")
        st.success(f"✅ {quantity} kg of {food_name} added!")

# Function to log food consumption
def log_consumption():
    st.subheader("🍽️ Log Food Consumption")
    food_name = st.text_input("Enter food name:")
    quantity = st.number_input("Enter weight consumed in kg:", min_value=0.1, step=0.1)
    consumption_date = st.date_input("Enter consumption date:", datetime.date.today())

    if st.button("Log Consumption"):
        with open(consumption_file, "a") as file:
            file.write(f"{food_name},{quantity},{consumption_date}\n")
        st.success(f"✅ {quantity} kg of {food_name} consumed!")

# Function to predict food surplus
def predict_surplus():
    st.subheader("📈 Food Surplus Prediction")

    try:
        purchases = pd.read_csv(purchase_file, names=["Food", "Purchased", "Date"])
        purchases["Date"] = pd.to_datetime(purchases["Date"], errors='coerce')
        purchases = purchases.dropna(subset=["Date"])
        purchases["Days_Ago"] = (pd.Timestamp.today() - purchases["Date"]).dt.days

        consumption = pd.read_csv(consumption_file, names=["Food", "Consumed", "Date"])
        consumption["Date"] = pd.to_datetime(consumption["Date"], errors='coerce')
        consumption = consumption.dropna(subset=["Date"])
        consumption["Days_Ago"] = (pd.Timestamp.today() - consumption["Date"]).dt.days

        purchase_groups = purchases.groupby("Food")
        consumption_groups = consumption.groupby("Food")

        predictions = {}

        for food in set(purchase_groups.groups.keys()).intersection(set(consumption_groups.groups.keys())):
            purchase_data = purchase_groups.get_group(food)
            consumption_data = consumption_groups.get_group(food)

            if len(consumption_data) > 2 and len(purchase_data) > 2:
                X_purchase = np.array(purchase_data["Days_Ago"]).reshape(-1, 1)
                y_purchase = np.array(purchase_data["Purchased"])

                X_consumption = np.array(consumption_data["Days_Ago"]).reshape(-1, 1)
                y_consumption = np.array(consumption_data["Consumed"])

                purchase_model = LinearRegression()
                purchase_model.fit(X_purchase, y_purchase)
                predicted_purchase = max(0, purchase_model.predict([[0]])[0])

                consumption_model = LinearRegression()
                consumption_model.fit(X_consumption, y_consumption)
                predicted_consumption = max(0, consumption_model.predict([[0]])[0])

                estimated_surplus = max(0, predicted_purchase - predicted_consumption)
                predictions[food] = estimated_surplus

        if predictions:
            st.write("📢 Forecasted Surplus:")
            for food, surplus in predictions.items():
                if surplus > 0:
                    st.write(f"**{food}**: Estimated surplus = **{surplus:.2f} kg**")
                    st.warning("⚠️ Consider adjusting purchases or donating excess.")
        else:
            st.info("⚡ Not enough data to predict trends.")

    except FileNotFoundError:
        st.error("⚠️ Purchase or consumption data not found.")

# Main app layout
st.title("🍏 AI-Powered Food Tracker")

menu_option = st.sidebar.radio("Menu", ["Add Purchase", "Log Consumption", "Predict Surplus"])

if menu_option == "Add Purchase":
    add_purchase()
elif menu_option == "Log Consumption":
    log_consumption()
elif menu_option == "Predict Surplus":
    predict_surplus()

