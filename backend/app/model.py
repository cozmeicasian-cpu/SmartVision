probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
top3_idx = probs.argsort()[-3:][::-1]
top3 = [(CLASSES[i], float(probs[i])) for i in top3_idx]

# Convert Grad-CAM to base64
buffer = io.BytesIO()
heatmap_img.save(buffer, format="PNG")
heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

# Convert Grad-CAM to base64
buffer = io.BytesIO()
heatmap_img.save(buffer, format="PNG")
heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

# Identify top prediction and its confidence
predicted_class, confidence = top3[0]  # highest class + probability

def predict_image(image_bytes):
    # your model prediction steps...
    probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(CLASSES[i], float(probs[i])) for i in top3_idx]

    # Convert Grad-CAM to base64
    buffer = io.BytesIO()
    heatmap_img.save(buffer, format="PNG")
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Identify top prediction and confidence
    predicted_class, confidence = top3[0]

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": top3,
        "heatmap": heatmap_base64
    }
