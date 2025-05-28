import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

class FruitMaturityPredictor:
    def __init__(self, model_path='fruit_maturity_model.h5', info_path='model_info.json'):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = 224  # AlexNet input size
        
        # Load class names and confidence threshold
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
                self.class_names = info['class_names']
                self.confidence_threshold = info['confidence_threshold']
        else:
            # Default values if file doesn't exist
            self.class_names = ['Defect', 'Immature', 'Mature']
            self.confidence_threshold = 0.7
        
        # Mapping to user-friendly labels
        self.display_names = {
            'Defect': 'Defect',
            'Immature': 'Raw',
            'Mature': 'Ripe'
        }

    def preprocess_image(self, image_bytes):
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # Resize image
        image = image.resize((self.img_size, self.img_size))
        # Convert to array and normalize
        image_array = np.array(image) / 255.0
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def predict(self, image_bytes):
        # Preprocess the image
        processed_image = self.preprocess_image(image_bytes)
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Check if the confidence is too low, indicating it might not be a dragon fruit
        if confidence < self.confidence_threshold:
            result_class = "Other"
            display_class = "Not a Dragon Fruit"
        else:
            result_class = self.class_names[predicted_class_idx]
            display_class = self.display_names.get(result_class, result_class)
        
        return {
            'class': result_class,
            'display_name': display_class,
            'confidence': confidence,
            'is_dragon_fruit': confidence >= self.confidence_threshold,
            'predictions': {
                self.display_names.get(class_name, class_name): float(pred)
                for class_name, pred in zip(self.class_names, predictions[0])
            }
        } 