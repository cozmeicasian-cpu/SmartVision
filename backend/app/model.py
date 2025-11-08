import io
import base64
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# ✅ Define your class names
CLASSES = ["Abyssinian", "Bengal", "Persian", "Siamese", "Maine Coon"]

# ✅ Load your model once (adjust path if necessary)
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(CLASSES))
model.load_state_dict(torch.load("app/model_weights.pth", map_location="cpu"))
model.eval()

# ✅ Grad-CAM or heatmap stub (replace if you have your own)
def generate_heatmap(image):
    # Dummy placeholder — replace with your Grad-CAM logic if needed
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

    # 2️⃣ Make prediction
    outputs = model(img_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(CLASSES[i], float(probs[i])) for i in top3_idx]

    # 3️⃣ Generate heatmap (optional)
    heatmap_img = generate_heatmap(image)
    buffer = io.BytesIO()
    heatmap_img.save(buffer, format="PNG")
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # 4️⃣ Return structured result
    predicted_class, confidence = top3[0]

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": top3,
        "heatmap": heatmap_base64
    }
