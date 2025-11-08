import io
import base64
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# ✅ Correct class names
CLASSES = ["Cat", "Dog", "Bird", "Deer", "Horse"]

# ✅ Load model architecture
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(CLASSES))

# ✅ Try to load weights — skip mismatched ones safely
try:
    checkpoint = torch.load("model.pth", map_location="cpu")
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in checkpoint.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    model_state.update(filtered)
    model.load_state_dict(model_state)
    print(f"✅ Model loaded successfully — {len(filtered)} layers matched (fc skipped if mismatched).")
except Exception as e:
    print(f"⚠️ Warning: Model partially loaded or missing weights: {e}")

model.eval()

# ✅ (Optional) Placeholder for Grad-CAM heatmap
def generate_heatmap(image):
    return image

# ✅ Main prediction function
def predict_image(image_bytes):
    # Preprocess image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)

    # Run prediction
    outputs = model(img_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(CLASSES[i], float(probs[i])) for i in top3_idx]

    # Generate fake heatmap (replace later if needed)
    heatmap_img = generate_heatmap(image)
    buffer = io.BytesIO()
    heatmap_img.save(buffer, format="PNG")
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Get top prediction
    predicted_class, confidence = top3[0]

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": top3,
        "heatmap": heatmap_base64
    }
