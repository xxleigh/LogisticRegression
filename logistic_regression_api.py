from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import pymysql

logging.basicConfig(level=logging.DEBUG)

print("Starting Flask API...")

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS for all routes

# Create a logistic regression model (this is a placeholder; ideally, you should train it with real data)
model = LogisticRegression()

# Placeholder data for training (replace this with real training data)
X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
y_train = np.array([0, 1, 1, 0])

# Train the model
model.fit(X_train, y_train)

# Save the model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Database connection setup
def get_db_connection():
    return pymysql.connect(
        host='localhost',       # Database host
        user='root',            # Database user
        password='admin',       # Database password
        db='login',             # Database name
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

@app.route('/suggest_accreditor', methods=['POST'])
def suggest_accreditor():
    logging.debug("Request received for /suggest_accreditor")
    try:
        # Get the data from the request in JSON format
        data = request.get_json()
        logging.debug(f"Received data: {data}")

        # Extract the specialization from the JSON data
        specialization = data['specialization']
        logging.debug(f"Specialization: {specialization}")

        # Connect to the database
        connection = get_db_connection()

        try:
            with connection.cursor() as cursor:
                # Query to get accreditors with the specified specialization
                sql = "SELECT name, webmail, specialization, designation FROM user_accreditor WHERE specialization=%s"
                cursor.execute(sql, (specialization,))
                matching_accreditors = cursor.fetchall()
                logging.debug(f"Matching accreditors found: {matching_accreditors}")

                # If there are matching accreditors, return them as suggestions
                if matching_accreditors:
                    return jsonify({"suggested_accreditors": matching_accreditors})
                else:
                    return jsonify({"message": "No accreditors found with the specified specialization."})
        finally:
            connection.close()

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    logging.debug("Request received for /predict")
    try:
        # Get the data from the request in JSON format
        data = request.get_json()

        # Extract features from the JSON data (assuming features are passed in an array)
        features = data['features']

        # Convert features to a NumPy array for the model
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)
        probability = model.predict_proba(features_array)[0, 1]

        # Return the prediction as a JSON response
        response = {
            'prediction': int(prediction[0]),
            'probability': float(probability)
        }
        logging.debug(f"Prediction result: {response}")
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    logging.debug("Running Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)