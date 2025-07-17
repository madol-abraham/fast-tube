# app.py
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Weather Classification API",
    description="API for classifying weather conditions in images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    model = tf.keras.models.load_model('dreamers_Weather Model.h5')
    # Alternative models if the above fails
    # model = tf.keras.models.load_model('best_model.h5')
    # model = tf.keras.models.load_model('dreamers_Weather Model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define class names
CLASS_NAMES = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

# Image preprocessing function
def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((256, 256))  # Resize to match model input size
        image = np.array(image) / 255.0  # Normalize to [0,1]
        if image.shape[-1] != 3:  # Ensure 3 channels (RGB)
            image = image.convert('RGB')
            image = np.array(image) / 255.0
        return np.expand_dims(image, axis=0)  # Add batch dimension
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Weather Classification API", "status": "active"}

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")
    
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": {
                CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# For local testing
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
