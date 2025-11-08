from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.model import predict_image
import io
from PIL import Image
import base64
from fastapi.responses import JSONResponse




app = FastAPI()

# ✅ Allow both localhost (for testing) and Vercel domain
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://smart-vision-caou.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    prediction, heatmap_img = predict_image(image_bytes)
    buffered = io.BytesIO()
    heatmap_img.save(buffered, format="PNG")
    heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"prediction": prediction, "heatmap": heatmap_b64}

from fastapi.responses import JSONResponse

@app.get("/")
def root():
    return JSONResponse({"message": "🚀 SmartVision API is running!"})
