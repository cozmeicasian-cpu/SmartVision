import cv2
import io
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
try:
    model.load_state_dict(torch.load("model.pth", map_location=device))
except FileNotFoundError:
    print("⚠️ model.pth not found — using pretrained ResNet18 features only.")
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

CLASSES = ["plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def generate_gradcam(model, img_tensor, target_layer):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, pred_class].backward()

    # Remove hooks
    handle_f.remove()
    handle_b.remove()

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

    cam = cam.squeeze().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam


def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
    label = CLASSES[pred.item()]

    cam = generate_gradcam(model, img_tensor, model.layer4[-1])
    cam = np.uint8(255 * cam)

    # convert to heatmap overlay
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    orig = np.array(image.resize((224, 224)))
    overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)
    heatmap_img = Image.fromarray(overlay)

    return label, heatmap_img
