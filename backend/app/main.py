from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.model import predict_image
import io
from PIL import Image
import base64

app = FastAPI()

# ✅ CORS middleware goes HERE (right after app = FastAPI())
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "SmartVision API is running ✅"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    prediction, heatmap_img = predict_image(image_bytes)

    # Convert heatmap to base64 for React frontend
    buffered = io.BytesIO()
    heatmap_img.save(buffered, format="PNG")
    heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"prediction": prediction, "heatmap": heatmap_base64}
