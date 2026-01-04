from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from prediction import predict_image
import os

app = FastAPI(title="Action Recognition API")


origins = [
    "http://localhost:5500",  
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        
    allow_credentials=True,
    allow_methods=["*"],          
    allow_headers=["*"],          
)


frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
if not os.path.exists(frontend_path):
    raise FileNotFoundError(f"Frontend folder not found at {frontend_path}")

app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
def read_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        label = predict_image(image_bytes)
        return {"predicted_action": label}
    except Exception as e:
        return {"error": str(e)}
