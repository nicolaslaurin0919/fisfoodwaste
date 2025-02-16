import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# File paths
purchase_file = "/Users/nicolaslaurin/Documents/purchase_data.csv"
consumption_file = "/Users/nicolaslaurin/Documents/consumption_data.csv"

# Function to add food purchase with expiration date
def add_food():
    print("\n🛒 Add a Food Item")
    food_name = input("Enter food name: ")
    
    while True:
        try:
            quantity = float(input("Enter weight in kg: "))
            if quantity <= 0:
                print("❌ Quantity must be greater than 0.")
                continue
            break
        except ValueError:
            print("❌ Invalid input! Enter a number (e.g., 1.5).")

    purchase_date = input("Enter purchase date (YYYY-MM-DD): ")
    expiry_date = input("Enter expiration date (YYYY-MM-DD): ")

    with open(purchase_file, "a") as file:
        file.write(f"{food_name},{quantity},{purchase_date},{expiry_date}\n")

    print(f"✅ {quantity} kg of {food_name} added with expiry date {expiry_date}!")

# Function to log food consumption
def log_consumption():
    print("\n🍽️ Log Food Consumption")
    food_name = input("Enter food name: ")

    while True:
        try:
            quantity = float(input("Enter weight consumed in kg: "))
            if quantity <= 0:
                print("❌ Quantity must be greater than 0.")
                continue
            break
        except ValueError:
            print("❌ Invalid input! Enter a number (e.g., 1.5).")

    consumption_date = input("Enter consumption date (YYYY-MM-DD): ")

    with open(consumption_file, "a") as file:
        file.write(f"{food_name},{quantity},{consumption_date}\n")

    print(f"✅ {quantity} kg of {food_name} logged as consumed!")

# Function to predict future surplus using AI
def predict_surplus():
    print("\n📈 Predicting Future Surplus Using AI...")

    try:
        # Load purchase data
        purchases = pd.read_csv(purchase_file, names=["Food", "Purchased", "Date", "Expiry"])
        
        # Ensure date parsing uses explicit format (YYYY-MM-DD)
        purchases["Date"] = pd.to_datetime(purchases["Date"], format="%Y-%m-%d", errors='coerce')
        purchases["Expiry"] = pd.to_datetime(purchases["Expiry"], format="%Y-%m-%d", errors='coerce')
        purchases = purchases.dropna(subset=["Date", "Expiry"])

        # Calculate time differences
        purchases["Days_Ago"] = (pd.Timestamp.today() - purchases["Date"]).dt.days
        purchases["Days_To_Expire"] = (purchases["Expiry"] - pd.Timestamp.today()).dt.days

        # Remove expired food
        purchases = purchases[purchases["Days_To_Expire"] >= 0]

        # Load consumption data
        consumption = pd.read_csv(consumption_file, names=["Food", "Consumed", "Date"])
        consumption["Date"] = pd.to_datetime(consumption["Date"], format="%Y-%m-%d", errors='coerce')
        consumption = consumption.dropna(subset=["Date"])
        consumption["Days_Ago"] = (pd.Timestamp.today() - consumption["Date"]).dt.days

        # Group by food type
        purchase_groups = purchases.groupby("Food")
        consumption_groups = consumption.groupby("Food")

        predictions = {}

        for food in set(purchase_groups.groups.keys()).intersection(set(consumption_groups.groups.keys())):
            purchase_data = purchase_groups.get_group(food)
            consumption_data = consumption_groups.get_group(food)

            if len(consumption_data) > 2 and len(purchase_data) > 2:
                # Prepare training data
                X_purchase = np.array(purchase_data["Days_Ago"]).reshape(-1, 1)
                y_purchase = np.array(purchase_data["Purchased"])

                X_consumption = np.array(consumption_data["Days_Ago"]).reshape(-1, 1)
                y_consumption = np.array(consumption_data["Consumed"])

                # Train AI models
                purchase_model = LinearRegression()
                purchase_model.fit(X_purchase, y_purchase)
                predicted_purchase = max(0, purchase_model.predict([[0]])[0])

                consumption_model = LinearRegression()
                consumption_model.fit(X_consumption, y_consumption)
                predicted_consumption = max(0, consumption_model.predict([[0]])[0])

                # Account for soon-to-expire food
                soon_expiring_food = purchases[purchases["Days_To_Expire"] <= 3]["Purchased"].sum()
                estimated_surplus = max(0, predicted_purchase - predicted_consumption + soon_expiring_food)

                predictions[food] = estimated_surplus

        # Display predictions
        if predictions:
            print("\n📢 Forecasted Surplus Items:")
            for food, surplus in predictions.items():
                if surplus > 0:
                    print(f"{food}: Estimated surplus = {surplus:.2f} kg")
                    print("⚠️ Consider adjusting purchases or donating excess.")
        else:
            print("⚡ Not enough data to predict trends.")

    except FileNotFoundError:
        print("⚠️ Purchase or consumption data not found.")

# Function to check simple food surplus without AI (for food expiring in more than 3 days)
def check_simple_surplus():
    print("\n📦 Checking Simple Food Surplus (Expiring in More Than 3 Days)...")
    
    try:
        df_purchases = pd.read_csv(purchase_file, names=["Food", "Purchased", "Date", "Expiry"])
        df_purchases["Expiry"] = pd.to_datetime(df_purchases["Expiry"], errors="coerce")
        df_purchases = df_purchases.dropna(subset=["Expiry"])
        
        # Filter for food that expires in more than 3 days
        surplus_food = df_purchases[df_purchases["Expiry"] > (pd.Timestamp.today() + pd.Timedelta(days=3))]
        
        if surplus_food.empty:
            print("✅ No surplus food with more than 3 days left.")
        else:
            print("\n🔹 Surplus Food (Expiring in More Than 3 Days):")
            print(surplus_food.groupby("Food")["Purchased"].sum().reset_index())
    
    except FileNotFoundError:
        print("⚠️ Purchase data file not found. Add food purchases first.")

# Updated menu function
def menu():
    while True:
        print("\n🍏 Food Tracker Menu:")
        print("1️⃣ Add a food item")
        print("2️⃣ Log food consumption")
        print("3️⃣ Predict future surplus using AI")
        print("4️⃣ Check simple surplus (Expiring in More Than 3 Days)")
        print("5️⃣ Exit")

        choice = input("Choose an option (1-5): ")

        if choice == "1":
            add_food()
        elif choice == "2":
            log_consumption()
        elif choice == "3":
            predict_surplus()
        elif choice == "4":
            check_simple_surplus()
        elif choice == "5":
            print("👋 Exiting program. Have a great day!")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1-5.")

# Start the program
menu()

