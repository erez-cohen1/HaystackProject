import cProfile

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import torch
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor

# should be uncommented
# Initialize CLIP model (once)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

room_labels = ["bedroom", "kitchen", "bathroom", "living room", "dining room", "balcony","not a room","exterior of house","view from window","boat"]


def analyze_image_batch(url_list):
    """Process a batch of image URLs according to spec"""
    images = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        images = list(executor.map(download_image, url_list))
    # 2. Process valid images with CLIP (batched)
    valid_images = [img for img in images if img is not None]
    room_probs = [None] * len(images)

    if valid_images:
        # Batch process all valid images
        inputs = clip_processor(
            text=room_labels,
            images=valid_images,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            room_probs_batch = logits_per_image.softmax(dim=1).cpu().numpy()

        # Map back to original positions
        room_probs = []
        img_idx = 0
        for img in images:
            if img is None:
                room_probs.append(None)
            else:
                room_probs.append(room_probs_batch[img_idx])
                img_idx += 1
    # # Add after CLIP processing
    # if torch.cuda.is_available():
    #     print(f"\nGPU Memory: Allocated {torch.cuda.memory_allocated() / 1e6:.1f}MB | "
    #           f"Cached {torch.cuda.memory_reserved() / 1e6:.1f}MB")
    #     print(f"GPU Usage: {torch.cuda.utilization()}%")
    # 3. Feature analysis with CLIP results
    results = []
    for img, probs in zip(images, room_probs):
        if img is None:
            results.append(None)
            continue

        np_img = np.array(img)
        height, width = np_img.shape[:2]
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        # edges = cv2.Canny(gray, 100, 200)
        # edges = cv2.Canny(gray, 50, 150)

        # New direct hue calculation
        hsv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
        avg_hue = np.mean(hsv_img[:, :, 0])

        # Add CLIP room type prediction
        room_type = room_labels[np.argmax(probs)] if probs is not None else "unknown"
        room_confidence = np.max(probs) if probs is not None else 0.0

        results.append({
            'resolution': (width, height),
            'brightness': float(np.mean(gray)),
            # 'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            # 'clutter_score': calculate_clutter(img),
            'mood': "warm" if (avg_hue < 50 or avg_hue > 330) else "cool",
            'room_type': room_type,
            'room_confidence': float(room_confidence)
        })

    return results

def download_image(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"⚠️ Failed to download {url}: {str(e)}")
        return None


# def calculate_clutter(img):
#     np_img = np.array(img)
#     gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
#
#     # 1. Edge Density (existing approach)
#     edges = cv2.Canny(gray, 100, 200)
#     edge_pct = np.sum(edges > 0) / edges.size
#
#     # 2. Object Count (new)
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     obj_count = len([c for c in contours if cv2.contourArea(c) > 100])  # Min 100px area
#
#     # 3. Color Variance (new)
#     color_std = np.std(np_img, axis=(0, 1)).mean()  # Higher = more color variation
#
#     # Combined score (adjust weights as needed)
#     clutter_score = 0.4 * edge_pct + 0.4 * (obj_count / 10) + 0.2 * (color_std / 50)
#     return float(clutter_score * 100)  # Convert to percentage

