probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
top3_idx = probs.argsort()[-3:][::-1]
top3 = [(CLASSES[i], float(probs[i])) for i in top3_idx]

# Convert Grad-CAM to base64
buffer = io.BytesIO()
heatmap_img.save(buffer, format="PNG")
heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

return {
    "prediction": predicted_class,
    "confidence": float(probabilities[top_idx]),  # e.g. 0.87
    "probabilities": list(zip(class_names, probabilities.tolist())),
    "heatmap": heatmap_base64
}
