import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import torch
from transformers import CLIPProcessor, CLIPModel

# --------------------------
# Load image
# --------------------------
url = "https://a0.muscache.com/pictures/miso/Hosting-1158063164891809038/original/d3ff00e7-fba6-45d9-af5b-a7e06876ce0e.jpeg"
response = requests.get(url)
img_pil = Image.open(BytesIO(response.content)).convert("RGB")
np_img = np.array(img_pil)

# --------------------------
# 1. Quality metrics
# --------------------------
height, width = np_img.shape[:2]
resolution = (width, height)
brightness = np.mean(np_img)
gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

print(f"Resolution: {resolution}")
print(f"Brightness: {brightness:.2f}")
print(f"Sharpness: {sharpness:.2f}")

# --------------------------
# 2. Clutter score (edges)
# --------------------------
edges = cv2.Canny(gray, 100, 200)
clutter_score = np.sum(edges > 0) / edges.size * 100  # percentage of edge pixels
print(f"Clutter score (edge density %): {clutter_score:.2f}")

# --------------------------
# 3. Color palette & mood
# --------------------------
pixels = np_img.reshape(-1, 3)
kmeans = KMeans(n_clusters=5, random_state=42).fit(pixels)
palette = np.uint8(kmeans.cluster_centers_)

# Convert to HSV for mood detection
hsv_palette = cv2.cvtColor(palette.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
avg_hue = np.mean(hsv_palette[:, 0])
mood = "Warm" if (avg_hue < 50 or avg_hue > 330) else "Cool"

print(f"Dominant colors (RGB): {palette}")
print(f"Mood: {mood}")

# --------------------------
# 4. Room classification (CLIP)
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

room_labels = ["bedroom", "kitchen", "bathroom", "living room", "dining room", "balcony"]

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
print("Predicted room type:", predicted_room)