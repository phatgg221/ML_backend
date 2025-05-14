
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

try:
    loaded_best_model = load_model("best_vgg_model.h5")
    print(f"Model loaded successfully from {"best_vgg_model.h5"}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# def preprocess_image(image_bytes):
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     image = image.resize((128, 128))  # Change size if needed
#     img_array = np.array(image) / 255.0
#     return np.expand_dims(img_array, axis=0)

@app.get("/get")
def read_root():
    if loaded_best_model:
        return {"Hello": "World"}
    else:
        return {"nothing to show"}
    

# @app.get("/predict")
# async def predict():
#     return {"prediction"}
