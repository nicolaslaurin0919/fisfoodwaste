import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

# File paths
purchase_file = "/Users/nicolaslaurin/Documents/purchase_data.csv"
food_icon_path = "/Users/nicolaslaurin/Documents/food_icons/"

# Load purchase data
df = pd.read_csv(purchase_file)
df["Purchased"] = pd.to_numeric(df["Purchased"], errors="coerce")

# Group data by food item
surplus = df.groupby("Food")["Purchased"].sum()

# Check if surplus contains valid numeric values
if surplus.empty or surplus.sum() == 0:
    print("⚠️ No surplus available to visualize.")
else:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart
    bars = surplus.plot(kind="bar", ax=ax, color="green")
    plt.xlabel("Food Item")
    plt.ylabel("Quantity (kg)")
    plt.title("Available Surplus by Food Type")

    # Add food icons to bars
    for i, bar in enumerate(bars.patches):
        food_name = surplus.index[i].lower()
        icon_file = os.path.join(food_icon_path, f"{food_name}.png")

        if os.path.exists(icon_file):
            img = Image.open(icon_file)
            img = img.resize((40, 40))  # Resize icon for better visibility
            x_offset = bar.get_x() + bar.get_width() / 2 - 0.1
            y_offset = bar.get_height() + 0.5
            ax.imshow(img, extent=[x_offset, x_offset + 0.2, y_offset, y_offset + 1], aspect="auto")

    # Show the chart
    plt.show()

