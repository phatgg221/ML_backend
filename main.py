from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from PIL import Image as Img
import numpy as np
import io
import requests
import aiohttp
import os
from IPython.display import Image
from io import StringIO

app = FastAPI()


origins = [
    "http://localhost:8080",
    "https://image-result-seeker-site.vercel.app/",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow the listed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Disease classes (from the image)
DISEASE_CLASSES = [
    "bacterial_leaf_blight", "bacterial_leaf_streak", "bacterial_panicle_blight",
    "blast", "brown_spot", "dead_heart", "downy_mildew", "hispa", "normal", "tungro"
]

# Paddy varieties (from the image)
PADDY_VARIETIES = [
    "ADT45", "IR20", "KarnatakaPonni", "Onthanel", "Ponni", 
    "Surya", "Zonal", "AndraPonni", "AtchayaPonni", "RR"
]

# Load models
try:
    # Disease classifier model (VGG)
    disease_model = load_model("best_vgg_model.h5")
    print("Disease classifier model loaded successfully")
    
    # Age regression model (MobileNetV3Small)
    age_model = load_model("MobileNetV3Small_model.keras")
    print("Age regression model loaded successfully")
    
    # Paddy variety classifier model (EfficientNet)
    variety_model = load_model("efficientnetb0_paddy_classifier.keras")
    print("Paddy variety classifier model loaded successfully")
    
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    models_loaded = False
    exit()

# Preprocess the image before feeding it to the model
def preprocess_image(image_bytes, target_size=(128, 128)):
    try:
        # For image bytes from aiohttp download
        if isinstance(image_bytes, bytes):
            img = Img.open(io.BytesIO(image_bytes))
        # For image URL
        else:
            response = requests.get(image_bytes)
            img = Img.open(io.BytesIO(response.content))
        
        # Resize to match model's expected input size
        img = img.resize(target_size)
        
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
    if models_loaded:
        return {"message": "All models loaded successfully!"}
    else:
        return {"message": "Models not loaded correctly"}

# Get available models endpoint
@app.get("/models")
def get_models():
    return {
        "available_models": [
            {"name": "disease_classifier", "type": "classification", "classes": DISEASE_CLASSES},
            {"name": "age_regression", "type": "regression", "output": "age in days"},
            {"name": "variety_classifier", "type": "classification", "classes": PADDY_VARIETIES}
        ]
    }

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
        
        # Process for disease classification (VGG model)
        disease_input = preprocess_image(image_bytes, target_size=(64, 64))
        disease_prediction = disease_model.predict(disease_input)
        disease_class_index = np.argmax(disease_prediction[0])
        disease_confidence = float(disease_prediction[0][disease_class_index])
        disease_class_name = DISEASE_CLASSES[disease_class_index]
        
        # Process for age regression (MobileNetV3Small)
        age_input = preprocess_image(image_bytes, target_size=(64, 64))
        age_prediction = age_model.predict(age_input)
        predicted_age = float(age_prediction[0][0])
        
        # Process for paddy variety classification (EfficientNet)
        variety_input = preprocess_image(image_bytes, target_size=(64, 64))
        variety_prediction = variety_model.predict(variety_input)
        variety_class_index = np.argmax(variety_prediction[0])
        variety_confidence = float(variety_prediction[0][variety_class_index])
        variety_class_name = PADDY_VARIETIES[variety_class_index]
        
        # Return combined results
        result = {
            "image_url": image_url,
            "disease_classification": {
                "class": disease_class_name,
                "class_index": int(disease_class_index),
                "confidence": disease_confidence,
                "description": get_disease_description(disease_class_name)
            },
            "age_regression": {
                "predicted_age_days": predicted_age,
                "estimated_planting_date": f"Approximately {int(predicted_age)} days old"
            },
            "variety_classification": {
                "variety": variety_class_name,
                "variety_index": int(variety_class_index),
                "confidence": variety_confidence,
                "description": get_variety_description(variety_class_name)
            }
        }
        
        return result
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in prediction: {e}")
        print(error_details)
        return {"error": str(e), "details": error_details}

# Helper function to get disease descriptions
def get_disease_description(disease_name):
    descriptions = {
        "bacterial_leaf_blight": "The bacterial disease results in dark lesions with yellow borders that cause leaf tissue death",
        "bacterial_leaf_streak": "The bacterial infection results in leaf tissue death through the formation of streaks or lesions",
        "bacterial_panicle_blight": "The bacterial disease targets rice panicle structures by killing flower clusters which leads to decreased grain yields",
        "blast": "Blast is a fungal disease that causes dark lesions on leaves, necks, and panicles of rice, and can lead to plant death",
        "brown_spot": "The disease produces brown circular lesions with yellowish margins on rice leaves which decrease photosynthesis and growth",
        "dead_heart": "Stem borer infestation leads to this condition when the insect pests damage the plant's growing point which results in stunted growth or plant death",
        "downy_mildew": "A fungal disease that causes yellowing of leaves with white fungal growth on the undersides, often reducing plant vigor",
        "hispa": "The hispa beetle infestation results in damage to rice plants through leaf consumption which leads to yield reduction",
        "normal": "This refers to healthy plants without any visible signs of disease or pest damage",
        "tungro": "A viral disease transmitted by leafhoppers that causes yellowing and stunting of rice plants, leading to yield loss"
    }
    return descriptions.get(disease_name, "No description available")

# Helper function to get variety descriptions
def get_variety_description(variety_name):
    descriptions = {
        "ADT45": "ADT45 is a popular high yielding variety of rice from Tamil Nadu (India), which is known for its excellent grain quality and good resistance to pests and diseases",
        "IR20": "IR20 is a widely grown rice type developed by the International Rice Research Institute, known for its resistance to drought and its good yield",
        "KarnatakaPonni": "KarnatakaPonni is a traditional rice variety grown in the southern region of India, particularly known for its taste and aroma",
        "Onthanel": "Onthanel is a variety known for its good cooking qualities and higher resistance to pests, widely cultivated in some parts of India",
        "Ponni": "Ponni is one of the most famous rice varieties in India, renowned for its smooth texture and aroma, widely used in South Indian cuisine",
        "Surya": "Surya is a high-yielding variety known for its long grain and excellent cooking quality, commonly grown in Indian rice fields",
        "Zonal": "Zonal is a rice variety developed for specific climatic zones, offering adaptability to different growing conditions",
        "AndraPonni": "AndraPonni is a popular variety of rice from Andhra Pradesh, known for its fine texture and aroma",
        "AtchayaPonni": "AtchayaPonni is another aromatic and high-yielding variety, often used in South Indian cooking due to its distinct fragrance and taste",
        "RR": "RR is a high-yielding variety of rice developed for various agro-climatic conditions, known for its disease resistance and good grain quality"
    }
    return descriptions.get(variety_name, "No description available")