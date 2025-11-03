# app/gradcam.py
import torch
import numpy as np
import cv2

def generate_gradcam(model, img_tensor, target_layer_name="layer4"):
    """
    Generates a Grad-CAM heatmap for the top predicted class.
    """
    model.eval()
    gradients = []
    activations = []

    # Hook for gradients and activations
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    # Register hooks on target layer
    target_layer = dict([*model.named_modules()])[target_layer_name]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(img_tensor)
    class_idx = torch.argmax(output, dim=1).item()

    # Backward pass on top predicted class
    model.zero_grad()
    output[0, class_idx].backward()

    # Extract grad and activations
    grad = gradients[0]
    act = activations[0]
    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

    # Normalize heatmap 0–1
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (224, 224))

    return cam
