import io
import base64
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# ✅ Load model architecture (ResNet18)
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features

# ✅ Try loading weights
try:
    checkpoint = torch.load("model.pth", map_location="cpu")

    # Detect number of output classes automatically
    if "fc.weight" in checkpoint:
        num_classes = checkpoint["fc.weight"].shape[0]
        print(f"✅ Detected {num_classes} output classes from checkpoint.")
    else:
        num_classes = 5  # fallback
        print("⚠️ Could not detect output classes — defaulting to 5.")
except Exception as e:
    num_classes = 5
    checkpoint = None
    print(f"⚠️ Warning loading model: {e} — defaulting to {num_classes} classes.")

# ✅ Build classifier dynamically
model.fc = torch.nn.Linear(num_features, num_classes)

# ✅ Load matching weights only (ignore mismatched fc)
if checkpoint is not None:
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in checkpoint.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    model_state.update(filtered)
    model.load_state_dict(model_state)
    print(f"✅ Model loaded successfully — {len(filtered)} layers matched (fc skipped if mismatched).")

model.eval()

# ✅ Auto-generate placeholder class names
CLASSES = [f"Class_{i}" for i in range(num_classes)]
print(f"🧩 Using class labels: {CLASSES}")

# ✅ Dummy Grad-CAM (placeholder)
def generate_heatmap(image):
    return image

# ✅ Main prediction function
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().detach().numpy()

    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(CLASSES[i], float(probs[i])) for i in top3_idx]
    predicted_class, confidence = top3[0]

    # Heatmap to base64
    heatmap_img = generate_heatmap(image)
    buffer = io.BytesIO()
    heatmap_img.save(buffer, format="PNG")
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": top3,
        "heatmap": heatmap_base64
    }
