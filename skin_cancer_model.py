
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os


MODEL_PATH = "/content/drive/MyDrive/Multiple Disease Detection Project/ML/Models/skin_cancer.keras"
CLASS_NAMES = ['Benign', 'Malignant']



def load_skin_cancer_model():
    """Load the trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")
    print(f"✅ Loading Skin Cancer Model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    return model



def preprocess_image(image_path):
   
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image file not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array



def predict_skin_cancer(image_path):
    """
    Returns:
        class_label (str): Predicted class
        confidence (float): Model confidence (0–1)
    """
    model = load_skin_cancer_model()
    img_array = preprocess_image(image_path)

    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])
    predicted_class = CLASS_NAMES[int(confidence > 0.5)]

    print(f"🧠 Prediction: {predicted_class} (Confidence: {confidence:.4f})")
    return predicted_class, confidence



def evaluate_skin_cancer_model(test_generator):
    """
    Evaluate model performance using a given test generator.
    Args:
        test_generator: Keras ImageDataGenerator for test data
    """
    model = load_skin_cancer_model()
    loss, accuracy = model.evaluate(test_generator)
    print(f"✅ Test Accuracy: {accuracy * 100:.2f}% | Loss: {loss:.4f}")
    return accuracy, loss

