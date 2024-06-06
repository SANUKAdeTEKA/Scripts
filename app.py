from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load the saved model using TFSMLayer
model_path = os.path.join(os.getcwd(), 'vehicle_damage_detection_model')
model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

# Define the classes for vehicle damage
classes = ["crack", "dent", "glass shatter", "lamp broken", "scratch", "tire flat"]

def preprocess_image(image):
    img = Image.open(io.BytesIO(image)).convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)  # ResNet50 typical input size
    inp_numpy = np.array(img)[None]
    inp = tf.constant(inp_numpy, dtype='float32')
    inp = inp / 255.0  # Normalize the image
    return inp

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            image = file.read()
            inp = preprocess_image(image)
            class_scores = model(inp)[0].numpy()  # Ensure this line is compatible with TFSMLayer
            predicted_class = classes[np.argmax(class_scores)]

            # For demonstration, fixed costs
            repair_costs = {
                "crack": 300,
                "dent": 500,
                "glass shatter": 400,
                "lamp broken": 200,
                "scratch": 150,
                "tire flat": 100
            }

            cost = repair_costs[predicted_class]

            response = {
                'predicted_class': predicted_class,
                'repair_cost': cost
            }

            return jsonify(response)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
