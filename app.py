import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = tf.keras.models.load_model('./Food.keras')


# Define class names
CLASS_NAMES = ['Besan_cheela', 'Biryani', 'Chapathi', 'Chole_bature', 'Dahl', 'Dhokla', 'Dosa', 'Gulab_jamun',
               'Idli', 'Jalebi', 'Kadai_paneer', 'Naan', 'Paani_puri', 'Pakoda', 'Pav_bhaji', 'Poha', 'Rolls', 'Samosa',
               'Vada_pav', 'chicken_curry', 'chicken_wings', 'donuts', 'fried_rice', 'grilled_salmon', 'hamburger',
               'ice_cream', 'not_food', 'pizza', 'ramen', 'steak', 'sushi']

def load_and_prep_image(image_path, img_shape=224, scale=False):
    """
    Load and preprocess image for model prediction
    
    Args:
        image_path (str): Path to the image file
        img_shape (int): Target image size
        scale (bool): Whether to scale pixel values to [0,1]
    
    Returns:
        Preprocessed image tensor
    """
    try:
        # Read image file
        img = tf.io.read_file(image_path)
        
        # Decode image 
        img = tf.image.decode_image(img, channels=3)
        
        # Resize image
        img = tf.image.resize(img, [img_shape, img_shape])
        
        # Optional scaling
        if scale:
            img = img / 255.0
        
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

@app.route('/', methods=['GET'])
def home():
    """Home route to check if API is running"""
    return "Keras Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the class of an uploaded image
    
    Returns:
        JSON response with prediction and confidence
    """
    try:
        # Check if image is present in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
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
        
        return jsonify({
            'prediction': predicted_class, 
            'confidence': confidence_score
        })
    
    except Exception as e:
        # Log the full error for debugging
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
