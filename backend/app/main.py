from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.model import predict_image
import io
import base64

app = FastAPI()

# ✅ Allow both localhost (for testing) and your deployed frontend
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
    # 👇 predict_image now returns a dict (prediction, confidence, heatmap, etc.)
    result = predict_image(image_bytes)
    return result

@app.get("/")
def root():
    return JSONResponse({"message": "🚀 SmartVision API is running!"})
