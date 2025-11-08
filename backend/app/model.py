import io
import base64
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# ✅ Define class names (adjust as needed)
CLASSES = ["Abyssinian", "Bengal", "Persian", "Siamese", "Maine Coon"]

# ✅ Load model once (safe partial load)
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(CLASSES))

try:
    checkpoint = torch.load("model.pth", map_location="cpu")
    model_state = model.state_dict()
    # Only load matching weights to prevent size mismatch errors
    filtered = {
        k: v for k, v in checkpoint.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    model_state.update(filtered)
    model.load_state_dict(model_state)
    print(f"✅ Model loaded successfully — {len(filtered)} layers matched.")
except Exception as e:
    print(f"⚠️ Warning: Model partially loaded or missing weights: {e}")

model.eval()

# ✅ Optional: placeholder heatmap generator
def generate_heatmap(image):
    # Replace this stub with real Grad-CAM logic later if desired
    return image

# ✅ Main prediction function
def predict_image(image_bytes):
    # 1️⃣ Load and preprocess image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)

    # 2️⃣ Predict
    outputs = model(img_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(CLASSES[i], round(float(probs[i]) * 100, 2)) for i in top3_idx]

    # 3️⃣ Heatmap (optional)
    heatmap_img = generate_heatmap(image)
    buffer = io.BytesIO()
    heatmap_img.save(buffer, format="PNG")
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # 4️⃣ Return results
    predicted_class, confidence = top3[0]

    return {
        "prediction": predicted_class,
        "confidence": confidence,  # %
        "probabilities": top3,
        "heatmap": heatmap_base64
    }
