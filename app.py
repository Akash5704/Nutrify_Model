import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = tf.keras.models.load_model('./Food.keras')
# Define MongoDB connection
client = MongoClient('mongodb+srv://jaiswarakash04:admin@cluster0.g2hng.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['Nutrify']
food_collection = db['Nutrients']

# Class names for predictions
CLASS_NAMES = [
    'Besan_cheela', 'Biryani', 'Chapathi', 'Chole_bature', 'Dahl', 'Dhokla', 
    'Dosa', 'Gulab_jamun', 'Idli', 'Jalebi', 'Kadai_paneer', 'Naan', 'Paani_puri', 
    'Pakoda', 'Pav_bhaji', 'Poha', 'Rolls', 'Samosa', 'Vada_pav', 'chicken_curry', 
    'chicken_wings', 'donuts', 'fried_rice', 'grilled_salmon', 'hamburger', 'ice_cream', 
    'not_food', 'pizza', 'ramen', 'steak', 'sushi'
]

def load_and_prep_image(image_path, img_shape=224, scale=False):
    """Load and preprocess image for model prediction"""
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, [img_shape, img_shape])
        if scale:
            img = img / 255.0
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

def get_nutrients(food_name, weight):
    """Fetch nutrients from MongoDB based on food name and weight"""
    try:
        print(f"Searching for food: {food_name}")
        food = food_collection.find_one({"name": {"$regex": f"^{food_name}$", "$options": "i"}})
        
        if not food:
            print(f"Food '{food_name}' not found in database.")
            return {"error": "Food not found"}
        
        # Calculate nutrient values based on weight and round to 2 decimal places
        nutrients = {
            "name": food["name"],
            "weight": weight,
            "calories": round(food["per_gram"]["calories"] * weight, 2),
            "protein": round(food["per_gram"]["protein"] * weight, 2),
            "carbs": round(food["per_gram"]["carbs"] * weight, 2),
            "fats": round(food["per_gram"]["fats"] * weight, 2),
        }
        return nutrients
    except Exception as e:
        print(f"Error fetching nutrients: {e}")
        return {"error": "Error fetching nutrient data from database"}

@app.route('/', methods=['GET'])
def home():
    """Home route to check if API is running"""
    return "Flask and MongoDB Nutrient API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """Predict food class and fetch nutrients"""
    try:
        # Check if image is present in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Get the weight from the request params
        # weight = float(request.args.get('weight', 100))  # Default weight is 100 if not provided
        weight = float(request.form.get('weight', 100))  # Correct way

        # Secure filename and save temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join("temp", filename)
        
        # Ensure temp directory exists
        os.makedirs("temp", exist_ok=True)
        
        # Save file
        file.save(filepath)
        
        # Validate file
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return jsonify({'error': 'Failed to save image'}), 400
        
        # Preprocess image
        img = load_and_prep_image(filepath)
        
        # Add batch dimension
        img = tf.expand_dims(img, axis=0)
        
        # Make prediction
        pred = model.predict(img)
        
        # Get predicted class and confidence
        predicted_class_index = np.argmax(pred[0])
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence_score = float(np.max(pred[0]))
        
        # Remove temporary file
        os.remove(filepath)
        
        # Fetch nutrient information from MongoDB
        nutrients = get_nutrients(predicted_class, weight)
        
        if "error" in nutrients:
            return jsonify(nutrients), 500
        
        return jsonify({
            'Food': predicted_class,
            'nutrients': nutrients
        })
    
    except Exception as e:
        # Log the full error for debugging
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

userfood_collection = db['users_foods']

@app.route('/add_food', methods=['POST'])
def add_food():
    data = request.json
    food_entry = {
        "food_name": data.get("food_name", "Unknown"),
        "weight": data.get("weight"),
        "calories": data.get("calories"),
        "protein": data.get("protein"),
        "carbs": data.get("carbs"),
    }
    userfood_collection.insert_one(food_entry)
    return jsonify({"message": "Food entry saved successfully!"}), 201

@app.route('/get_foods', methods=['GET'])
def get_foods():
    foods = list(userfood_collection.find({}, {"_id": 0}))  # Exclude MongoDB _id field
    return jsonify(foods), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
