from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained AI model
model = joblib.load("ai_pricing_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        # Get JSON data from request
        data = request.json
        demand = data.get("demand")
        supply = data.get("supply")
        seasonal_factor = data.get("seasonal_factor")
        competitor_price = data.get("competitor_price")

        # Validate input data
        if None in [demand, supply, seasonal_factor, competitor_price]:
            return jsonify({"error": "Missing required parameters"}), 400

        # Convert input to array for prediction
        input_data = np.array([[demand, supply, seasonal_factor, competitor_price]])
        predicted_price = model.predict(input_data)

        # Return predicted price
        return jsonify({"predicted_price": round(predicted_price[0], 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the API server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
