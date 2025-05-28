from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import sys
import os

# Add the model directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.predict import FruitMaturityPredictor

app = FastAPI(title="Dragon Fruit Maturity Detection API (AlexNet)")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the predictor with AlexNet model
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'model_output', 'fruit_maturity_model.h5')
info_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'model_output', 'model_info.json')
predictor = FruitMaturityPredictor(model_path, info_path)

@app.get("/")
async def root():
    return {
        "message": "Dragon Fruit Maturity Detection API",
        "model": "AlexNet",
        "endpoints": {
            "/predict": "POST - Upload an image for maturity detection",
            "/app": "GET - Access the web interface"
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the file
        contents = await file.read()
        
        # Make prediction
        result = predictor.predict(contents)
        
        # Add model information to response
        if 'error' not in result:
            result['model_info'] = {
                'type': 'AlexNet',
                'confidence_threshold': predictor.confidence_threshold
            }
        
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "model_type": "AlexNet"
            }
        )

# Serve the frontend
try:
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')
    app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    
    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend():
        with open(os.path.join(frontend_dir, "index.html"), "r") as f:
            return f.read()
except Exception as e:
    print(f"Warning: Could not mount frontend: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 