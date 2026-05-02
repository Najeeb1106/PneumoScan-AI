import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import keras
from PIL import Image

app = FastAPI(title="PneumoScan AI API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Manual Model Architecture Reconstruction
def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "model", "weights.weights.h5")
load_error = None
try:
    model = create_model()
    model.load_weights(WEIGHTS_PATH)
    print("Model and weights loaded successfully.")
except Exception as e:
    load_error = str(e)
    print(f"Error loading model: {load_error}")
    model = None

def preprocess_image(image_bytes):
    """Preprocess the image to match model input (150x150, normalized)."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.get("/", response_class=HTMLResponse)
async def read_home():
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard():
    with open(os.path.join(static_dir, "dashboard.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/features", response_class=HTMLResponse)
async def read_features():
    with open(os.path.join(static_dir, "features.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/how-it-works", response_class=HTMLResponse)
async def read_how_it_works():
    with open(os.path.join(static_dir, "how-it-works.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/about", response_class=HTMLResponse)
async def read_about():
    with open(os.path.join(static_dir, "about.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": f"Model not loaded on server. Details: {load_error}"}
    
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        
        prediction = model.predict(processed_image)
        score = float(prediction[0][0])
        
        # 0 = Normal, 1 = Pneumonia (based on standard class indices from flow_from_directory)
        # Assuming PNEUMONIA is class 1 (probability closer to 1)
        result = "Pneumonia Detected" if score > 0.5 else "Normal (Healthy Lungs)"
        confidence = score if score > 0.5 else (1 - score)
        
        return {
            "prediction": result,
            "confidence": f"{confidence * 100:.2f}%",
            "raw_score": score
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
