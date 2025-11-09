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
    try:
        image_bytes = await file.read()
        result = predict_image(image_bytes)
        return result
    except Exception as e:
        import traceback
        print("🔥 Prediction error:", traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/")
def root():
    return JSONResponse({"message": "🚀 SmartVision API is running!"})
