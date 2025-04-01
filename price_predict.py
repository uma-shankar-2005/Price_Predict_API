import joblib

# Load the trained model
model = joblib.load("ai_pricing_model.pkl")

# Example: Predict price for new data (demand=200, supply=180, seasonal_factor=1.5, competitor_price=50)
new_data = [[200, 180, 1.5, 50]]
predicted_price = model.predict(new_data)

print(f"Predicted Selling Price: ₹{predicted_price[0]:.2f}")
