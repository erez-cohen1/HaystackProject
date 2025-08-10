import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import torch
from transformers import CLIPProcessor, CLIPModel



# Load model & processor once
device = "cuda" if torch.cuda.is_available() else "cpu"

room_labels = ["bedroom", "kitchen", "bathroom", "living room", "dining room", "balcony","not a room","exterior of house","view from window","boat"]
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

def analyze_image(url):
    try:
        response = requests.get(url)  # optional timeout
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx, 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image from {url}: {e}")
        response = None
    img_pil = Image.open(BytesIO(response.content)).convert("RGB")
    np_img = np.array(img_pil)

    # Quality metrics
    height, width = np_img.shape[:2]
    resolution = (width, height)
    brightness = np.mean(np_img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Clutter score
    edges = cv2.Canny(gray, 100, 200)
    clutter_score = np.sum(edges > 0) / edges.size * 100

    # Color palette & mood
    pixels = np_img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5, random_state=42).fit(pixels)
    palette = np.uint8(kmeans.cluster_centers_)
    hsv_palette = cv2.cvtColor(palette.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    avg_hue = np.mean(hsv_palette[:, 0])
    mood = "Warm" if (avg_hue < 50 or avg_hue > 330) else "Cool"

    # Room classification
    inputs = processor(
        text=room_labels,
        images=img_pil,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    predicted_room = room_labels[probs.argmax()]

    # Return all metrics as dict
    return {
        "resolution": resolution,
        "brightness": brightness,
        "sharpness": sharpness,
        "clutter_score": clutter_score,
        "color_palette": palette,
        "mood": mood,
        "room_type": predicted_room,
        "room_confidence": probs.max().item()
    }

# # Example usage with multiple URLs:
# urls = [
#     "https://a0.muscache.com/pictures/miso/Hosting-1158063164891809038/original/d3ff00e7-fba6-45d9-af5b-a7e06876ce0e.jpeg"
# ]
#
# for url in urls:
#     results = analyze_image(url)
#     print(f"Results for {url}:\n", results)