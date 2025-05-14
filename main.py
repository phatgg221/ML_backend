from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from PIL import Image as Img
import numpy as np
import io
import requests
import aiohttp
from IPython.display import Image
from io import StringIO

app = FastAPI()

# CORS middleware to allow cross-origin requests
origins = [
    "http://localhost:8080",
    "https://image-result-seeker-site.vercel.app/",  # Add your actual frontend URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow the listed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load the model
try:
    loaded_best_model = load_model("MobileNetV3Small_model.keras")
    print(f"Model loaded successfully from 'best_vgg_model.h5'")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Preprocess the image before feeding it to the model
# Preprocess the image before feeding it to the model
def preprocess_image(image_bytes):
    try:
        # For image bytes from aiohttp download
        if isinstance(image_bytes, bytes):
            img = Img.open(io.BytesIO(image_bytes))
        # For image URL
        else:
            response = requests.get(image_bytes)
            img = Img.open(io.BytesIO(response.content))
        
        # Resize to match model's expected input size
        img = img.resize((128, 128))
        
        # Convert to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        input_tensor = np.expand_dims(img_array, axis=0)
        
        print(f"Input tensor shape: {input_tensor.shape}")  # Debugging
        return input_tensor
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise e

# Create a Pydantic model to parse the incoming request body
class ImageRequest(BaseModel):
    image_url: str

# Route to check if the server is working
@app.get("/")
def read_root():
    if loaded_best_model:
        return {"message": "Model loaded successfully!"}
    else:
        return {"message": "Model not loaded"}

# Endpoint to handle image URL and prediction
@app.post("/predict")
async def predict(req: ImageRequest):
    try:
        image_url = req.image_url  # Get the image URL from the request
        print(f"Processing URL: {image_url}")
        
        # Download the image from the URL
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status != 200:
                    return {"error": f"Failed to fetch the image from the URL. Status: {response.status}"}
                image_bytes = await response.read()
        
        # Preprocess the image using the bytes directly
        input_tensor = preprocess_image(image_bytes)
        
        # Make prediction
        prediction = loaded_best_model.predict(input_tensor)
        
        # Find the predicted class index and confidence
        pred_class_index = np.argmax(prediction[0])
        confidence = float(prediction[0][pred_class_index])
        
        # Return prediction results
        result = {
            "prediction": prediction.tolist(),
            "predicted_class": int(pred_class_index),
            "confidence": confidence,
            "image_url": image_url
        }
        
        return result
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in prediction: {e}")
        print(error_details)
        return {"error": str(e), "details": error_details}

