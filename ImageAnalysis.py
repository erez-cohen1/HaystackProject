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

# Initialize CLIP model (once)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

room_labels = ["bedroom", "kitchen", "bathroom", "living room", "dining room", "balcony","not a room","exterior of house","view from window","boat"]


def analyze_image_batch(url_list):
    """Process a batch of image URLs according to spec"""
    images = []

    # 1. Download images (with error handling)
    # for url in url_list:
    #     try:
    #         response = requests.get(url, timeout=5)
    #         response.raise_for_status()  # Raise error for bad status
    #         img = Image.open(BytesIO(response.content)).convert("RGB")
    #         images.append(img)
    #     except Exception as e:
    #         print(f"⚠️ Failed to download {url}: {str(e)}")
    #         images.append(None)
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
        edges = cv2.Canny(gray, 100, 200)

        # New direct hue calculation
        hsv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
        avg_hue = np.mean(hsv_img[:, :, 0])

        # Add CLIP room type prediction
        room_type = room_labels[np.argmax(probs)] if probs is not None else "unknown"
        room_confidence = np.max(probs) if probs is not None else 0.0

        results.append({
            'resolution': (width, height),
            'brightness': float(np.mean(gray)),
            'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            'clutter_score': float(np.sum(edges > 0) / edges.size * 100),
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




