import numpy as np
from tensorflow.keras.models import load_model

# Define the normalization function that the Lambda layer likely uses
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def normalize(x):
    return (x - mean) / std

try:
    model = load_model('d3net_deployment_safe.keras', safe_mode=False, compile=False, custom_objects={'lambda': normalize})
    print("Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
except Exception as e:
    print(f"Error loading model: {e}")
