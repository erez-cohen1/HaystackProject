"""
Amenity Normalization Pipeline

This script processes Airbnb-style datasets to normalize amenity strings into
binary feature columns. It combines:
  - Text cleaning and stopword removal
  - Named entity recognition (Flair) for removing brand names
  - Semantic similarity classification using spaCy word vectors
  - Rule-based category matching
  - Output of a normalized CSV with binary amenity features
"""

import pandas as pd
import re
import ast
from tqdm import tqdm
from collections import defaultdict
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os

# --- Flair setup for Named Entity Recognition ---
try:
    from flair.data import Sentence
    from flair.models import SequenceTagger
    FLAIR_AVAILABLE = True
except ImportError:
    print("⚠ Flair not installed. Brand removal will be skipped.")
    FLAIR_AVAILABLE = False

# --- spaCy NLP Setup ---
try:
    NLP = spacy.load("en_core_web_md")
    print("spaCy NLP model (en_core_web_md) loaded successfully.")
except OSError:
    print("spaCy model 'en_core_web_md' not found.")
    print("   Please run: python -m spacy download en_core_web_md")
    exit()

# --- Stopwords (customized for amenities) ---
CUSTOM_STOP_WORDS = set(CountVectorizer(stop_words="english").get_stop_words()).union(
    ["listing", "extra", "cost", "included", "private"]
)

# --- Amenity Category Dictionaries ---

# Dictionary for rule-based matching
AMENITY_CATEGORIES = {
    "safety_core": ["smoke alarm", "carbon monoxide alarm", "fire extinguisher", "lock"],
    "safety_extra": ["first aid", "lockbox", "keypad", "safe", "guards"],
    "kitchen_core": ["kitchen", "refrigerator", "fridge", "oven", "stove", "cooktop", "microwave", "kettle",
                     "hot water kettle", "freezer"],
    "kitchen_extra": ["dishwasher", "blender", "baking", "toaster", "kitchenette", "rice maker"],
    "laundry_core": ["washer", "dryer", "washing machine", "tumble dryer", "iron", "drying rack"],
    "laundry_extra": ["laundromat"],
    "air_conditioner_core": ["air conditioning", "ac", "aircon"],
    "air_conditioner_extra": ["fan"],
    "heater_core": ["heating"],
    "heater_extra": ["heater", "fireplace"],
    "hygiene_core": ["shampoo", "soap", "towel", "shower", "hot water", "essentials", "conditioner", "body soap"],
    "hygiene_extra": ["bathtub", "bath", "bidet"],
    "wellness_core": ["sauna", "hot tub", "jacuzzi", "spa"],
    "storage_core": ["closet", "wardrobe", "dresser"],
    "storage_extra": ["hangers"],
    "entertainment_core": ["tv", "hdtv", "television"],
    "entertainment_extra": ["hbo", "cable", "netflix", "hulu", "amazon prime", "disney", "books", "games", "console",
                            "record player", "piano"],
    "internet_core": ["wifi", "internet", "wireless internet", "ethernet"],
    "parking_core": ["parking", "garage", "carport"],
    "parking_extra": ["ev charger"],
    "exercise_core": ["gym", "fitness", "exercise", "weights", "treadmill", "yoga mat", "ping pong table"],
    "exercise_extra": ["climbing wall"],
    "outdoors_core": ["patio", "balcony", "pool"],
    "outdoors_extra": ["grill", "bbq", "lounger", "lounge", "fire pit", "firepit", "sun lounger"],
    "children_care_core": ["crib", "high chair", "children", "baby", "changing table"],
    "children_care_extra": ["childrens dinnerware", "babysitter"],
    "cleaning_core": ["cleaning"],
    "cleaning_extra": ["housekeeping"],
    "pets_allowed": ["pets allowed"],
    "smoking_allowed": ["smoking allowed"],
    "coffee_maker_core": ["coffee maker", "coffee", "nespresso", "keurig"],
    "cooking_basics_core": ["cooking basics", "dinnerware", "table"],
    "sound_system_core": ["sound system", "sound", "speakers"],
    "accessibility_core": ["elevator", "lift", "wheelchair ramp", "ground floor"],
    "view_core": ["view", "waterfront"],
    "bedroom_core": ["bed", "pillow", "linens", "blanket", "duvet", "sheet"],
    "bedroom_extra": ["room darkening shades"],
    "attractions_nearby_core": ["restaurant", "coffee shop", "lake", "beach", "river", "center", "golf", "bowling",
                                "resort", "ski"],
    "attractions_nearby_extra": ["museum"],
    "hospitality_core": ["self check-in"],
    "hospitality_extra": ["host greets", "building staff", "breakfast"],
    "transport_core": ["bus stop", "bike", "boat", "bicycle", "train"],
    "transport_extra": ["kayak"]
}

# Semantic representatives (used for spaCy vector similarity)
CATEGORY_REPRESENTATIVES = {
    "exercise_core": ["gym", "exercise equipment", "weights", "treadmill", "yoga mat", "ping pong table"],
    "kitchen_core": ["kitchen", "appliance", "refrigerator", "oven", "stove", "fridge"],
    "safety_core": ["safety", "security", "alarm", "detector", "lock", "first aid"],
    "outdoors_core": ["outdoors", "backyard", "garden", "patio", "balcony", "grill", "bbq"],
    "cooking_basics_core": ["cooking basics", "dinnerware", "utensils", "cups", "glasses", "forks", "spoons", "pots"],
}

# --- Utility functions ---
def clean_amenity_text(text: str) -> str:
    """Lowercase, remove non-letters, and drop stopwords from amenity text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in CUSTOM_STOP_WORDS]
    return re.sub(r"\s+", " ", " ".join(words)).strip()


# --- Flair-based brand remover ---
class FlairBrandRemover:
    """Removes brand names (ORG, PER, MISC entities) from amenity text using Flair."""

    def __init__(self):
        self.tagger = None
        if FLAIR_AVAILABLE:
            try:
                self.tagger = SequenceTagger.load("ner-large")
                print("Flair NER model loaded.")
            except Exception as e:
                print(f"Failed to load Flair NER model: {e}")

    def remove_brands(self, text: str) -> str:
        if not self.tagger or not text:
            return text
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        brands_to_remove = {ent.text.lower() for ent in sentence.get_spans("ner")
                            if ent.tag in ["ORG", "PER", "MISC"]}
        if not brands_to_remove:
            return text
        return " ".join([w for w in text.split() if w.lower() not in brands_to_remove])


# --- Semantic classifier using spaCy vectors ---
class AmenityClassifier:
    """Classifies amenities into categories using semantic similarity with spaCy embeddings."""

    def __init__(self, category_definitions: dict):
        self.category_vectors = {}
        print("Calculating semantic vectors for representative categories...")
        for category, examples in tqdm(category_definitions.items(), desc="Vectorizing"):
            example_vecs = [NLP(ex).vector for ex in examples if NLP(ex).has_vector]
            if example_vecs:
                self.category_vectors[category] = np.mean(example_vecs, axis=0)

    def _cosine_similarity(self, v1, v2) -> float:
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def predict(self, text: str, threshold: float = 0.4):
        """Return best category if similarity > threshold, else None."""
        if not text or not NLP(text).has_vector:
            return None
        vec = NLP(text).vector
        best, max_sim = None, -1
        for cat, cat_vec in self.category_vectors.items():
            sim = self._cosine_similarity(vec, cat_vec)
            if sim > max_sim:
                max_sim, best = sim, cat
        return best if max_sim > threshold else None


# --- Core Processing Functions ---
def classify_single_amenity(raw: str, category_map, classifier, brand_remover):
    """Classify a single amenity by rule-based terms, then semantic similarity."""
    raw_lower = raw.lower()
    for cat, terms in category_map.items():
        for term in terms:
            if re.search(r"\b" + re.escape(term) + r"\b", raw_lower):
                return cat
    cleaned = clean_amenity_text(brand_remover.remove_brands(raw))
    return classifier.predict(cleaned)


def inspect_amenity_classification(df, classifier, brand_remover):
    """Classify all unique amenities in the dataset and group them by category."""
    all_unique = set(a for lst in df["parsed_amenities"] for a in lst)
    print(f"\nFound {len(all_unique)} unique amenities.")
    groups = defaultdict(list)
    for amenity in tqdm(sorted(all_unique), desc="Classifying"):
        cat = classify_single_amenity(amenity, AMENITY_CATEGORIES, classifier, brand_remover)
        groups[cat if cat else "__UNCLASSIFIED__"].append(amenity)
    return groups


def create_binary_columns(df, groups):
    """Convert classified amenities into binary one-hot columns per category."""
    print("\nCreating binary columns...")
    amenity_to_cat = {a: c for c, ams in groups.items() if c != "__UNCLASSIFIED__" for a in ams}
    categories = sorted([c for c in groups.keys() if c != "__UNCLASSIFIED__"])

    tqdm.pandas(desc="Mapping")
    df["amenity_categories"] = df["parsed_amenities"].progress_apply(
        lambda ams: {amenity_to_cat.get(a) for a in ams if amenity_to_cat.get(a)}
    )

    for cat in tqdm(categories, desc="Binary Columns"):
        df[f"amenity_{cat}"] = df["amenity_categories"].apply(lambda s: 1 if cat in s else 0)

    df = df.drop(columns=["parsed_amenities", "amenity_categories"])
    print("Binary columns created.")
    return df


def normalize_amenities(df, output_file="normalized_amenities.csv"):
    """Run full pipeline: parse amenities, classify, create binary features, save CSV."""
    print("\nNormalizing amenities...")

    classifier = AmenityClassifier(CATEGORY_REPRESENTATIVES)
    brand_remover = FlairBrandRemover()

    tqdm.pandas(desc="Parsing")
    df["parsed_amenities"] = df["amenities"].progress_apply(
        lambda s: ast.literal_eval(s.strip()) if isinstance(s, str) and s.strip().startswith("[") else []
    )

    groups = inspect_amenity_classification(df.copy(), classifier, brand_remover)

    print("\n--- Classification Review ---")
    for cat, ams in sorted(groups.items()):
        print(f"\n{cat}: {len(ams)} items")
        for a in ams[:10]:
            print(" -", a)
        if len(ams) > 10:
            print(f" ... and {len(ams) - 10} more")

    df_transformed = create_binary_columns(df, groups)

    print("\nPreview of new columns:")
    amenity_cols = sorted([c for c in df_transformed.columns if c.startswith("amenity_")])
    print(df_transformed[["id", "name"] + amenity_cols[:10]].head())

    print(f"\nSaving to {output_file}")
    df_transformed.to_csv(output_file, index=False)
    print("✓ File saved.")


# --- Main entrypoint ---
if __name__ == "__main__":
    df = pd.read_csv("clean_merged_database.csv")
    normalize_amenities(df, "normalized_amenities.csv")
