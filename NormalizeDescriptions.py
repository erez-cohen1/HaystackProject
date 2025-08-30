import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ------------------------
# CONFIGURATION
# ------------------------
USE_SEMANTIC_TAGGING = True
SEMANTIC_SIMILARITY_THRESHOLD = 0.4
MODEL_NAME = "all-MiniLM-L6-v2"       # Fast & accurate small model
TAGS = {
    "Desc_Transport": [
        "bus", "metro", "subway", "tube", "station", "train", "tram", "airport", "public transport",
        "direct connection", "easy access",
        "well-connected", "scooter", "bike path"
    ],
    "Desc_Attractions": [
        "museum", "gallery", "theatre", "zoo", "park", "beach", "lake", "river", "bars", "restaurants",
        "cafe", "nightlife", "entertainment", "shopping", "market", "historic", "landmarks", "city center",
        "downtown", "convention center"
    ],
    "Desc_Neighborhood": [
        "quiet", "safe", "lively", "central", "trendy", "residential", "popular", "peaceful", "charming",
        "vibrant", "up-and-coming", "family-friendly", "student area", "historic district", "a short distance from",
        "walking distance", "minutes away", "steps away", "close to"
    ],
    "Desc_Property_Type": [
        "apartment", "condo", "house", "villa", "loft", "studio", "bungalow", "cabin", "townhouse",
        "guesthouse", "private room", "entire place"
    ],
    "Desc_Space_Layout": [
        "large", "spacious", "small", "compact", "cozy", "open plan", "floor", "private entrance",
        "ensuite", "private bathroom", "shared bathroom"
    ],
    "Desc_Capacity": [
        "sleeps", "capacity", "people", "guests", "family", "couple", "solo", "group", "extra bed",
        "sofa bed"
    ],
    "Desc_Outdoor_Features": [
        "balcony", "terrace", "garden", "patio", "pool", "private pool", "shared pool", "bbq",
        "barbecue", "rooftop", "backyard", "deck", "jacuzzi"
    ],
    "Desc_View": [
        "view", "sea view", "ocean view", "city view", "skyline", "mountain view", "garden view",
        "panoramic", "breathtaking views"
    ],
    "Desc_Kitchen": [
        "kitchen", "kitchenette", "fridge", "freezer", "stove", "oven", "microwave", "dishwasher",
        "utensils", "cookware", "kettle", "coffee machine", "espresso", "Nespresso", "toaster",
        "blender", "fully equipped"
    ],
    "Desc_Living_room": [
        "couch", "sofa","table", "chair", "desk", "workspace", "dedicated workspace"
    ],
    "Desc_Entertainment": [
      "tv", "smart tv", "netflix", "hbo", "cable", "projector", "speakers", "sound system", "books", "games"
    ],
    "Desc_Connectivity": [
        "wifi", "internet", "high-speed", "fiber optic"
    ],
    "Desc_Climate": [
        "air conditioning", "ac", "heating","central heating", "fireplace", "fan"
    ],
    "Desc_Bedroom": [
        "bedroom", "bed", "king bed", "queen bed", "double bed", "single bed", "bunk bed", "crib", "sheets", "linen"

    ],
    "Desc_Bath": [
        "shower", "bathtub", "towels", "essentials", "shampoo", "soap", "hairdryer", "hot water"
    ],
    "Desc_Safety_Accessibility": [
        "first aid kit", "fire extinguisher", "smoke detector", "carbon monoxide detector", "elevator",
        "lift", "wheelchair accessible", "step-free access", "ground floor"
    ],
    "Desc_Atmosphere": [
        "cozy", "homey", "comfortable", "relaxing", "inviting", "warm", "charming", "modern", "stylish",
        "luxury", "designer", "rustic", "bohemian", "vintage", "minimalist", "newly renovated",
        "brand new", "updated", "classic"
    ],
    "Desc_Cleanliness": [
        "clean", "tidy", "spotless", "well-maintained", "fresh", "hygiene"
    ],
    "Desc_Check_in": [
        "check-in", "check-out", "flexible", "self check-in", "lockbox", "keypad", "smart lock",
        "24-hour check-in"
    ],
    "Desc_Host": [
        "host lives here", "host on site", "available", "recommendations", "local tips", "superhost"
    ],
    "Desc_Rules": [
        "smoking", "no smoking", "pets", "pets allowed", "no pets", "parties", "no parties", "events",
        "rules", "quiet hours"
    ],
    "Desc_Suitability": [
        "family-friendly", "kid-friendly", "business travel", "remote work", "couples",
        "romantic getaway", "long-term stays"
    ],
    "Desc_Fees": [
        "deposit", "fee", "cleaning fee", "extra guest fee"
    ]
}

# ------------------------
# LOAD DATA
# ------------------------
# try:
#     df = pd.read_csv("clean_merged_database.csv")
# except FileNotFoundError:
#     print("❌ listings.csv not found")
#     exit()
#
# if "description" not in df.columns:
#     raise ValueError("No 'description' column found in listings.csv")

# ------------------------
# LOAD MODEL & PRECOMPUTE TAG EMBEDDINGS
# ------------------------
model = SentenceTransformer(MODEL_NAME)

tag_names = list(TAGS.keys())
tag_descriptions = ["; ".join(words) for words in TAGS.values()]
tag_embeddings = model.encode(tag_descriptions, convert_to_tensor=True, normalize_embeddings=True)

# ------------------------
# TAGGING FUNCTION
# ------------------------
def tag_description(text):
    if not isinstance(text, str) or not text.strip():
        return "none"

    text_lower = text.lower()
    tags_found = set()

    # --- 1. Keyword Spotting (regex) ---
    # for tag, keywords in TAGS.items():
    #     for kw in keywords:
    #         # match plural/singular forms automatically
    #         pattern = rf"\b{re.escape(kw)}(s|es)?\b"
    #         if re.search(pattern, text_lower):
    #             tags_found.add(tag)
    #             break

    # --- 2. Semantic Matching (cosine similarity) ---
    if USE_SEMANTIC_TAGGING:
        # Split text into sentences for better granularity
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

        if sentences:  # Only proceed if non-empty sentences exist
            sent_embeddings = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
            cos_scores = util.cos_sim(sent_embeddings, tag_embeddings)  # shape: (num_sentences, num_tags)
            
            # Collect tags that exceed threshold
            for i in range(cos_scores.shape[0]):
                for j in range(cos_scores.shape[1]):
                    if cos_scores[i, j] >= SEMANTIC_SIMILARITY_THRESHOLD:
                        tags_found.add(tag_names[j])

    return ", ".join(sorted(tags_found)) if tags_found else "none"

def normalize_descriptions(df,outputfile='normalized_description.csv'):

    # ------------------------
    # APPLY TO DATAFRAME
    # ------------------------
    print("Tagging descriptions...")
    df["tags"] = df["description"].apply(tag_description)

    # ------------------------
    # SAVE TO CSV
    # ------------------------
    df[["id", "description", "tags"]].to_csv("tagged_listings.csv", index=False, encoding="utf-8")
    print("Tagged data saved to tagged_listings.csv")

    # ------------------------
    # CONVERT TAGS COLUMN TO BINARY ONE-HOT ENCODING
    # ------------------------

    # df = pd.read_csv('tagged_listings.csv')
    one_hot_df = df["tags"].str.get_dummies(sep=", ")

    # Add the ID column to the one-hot encoded dataframe
    one_hot_df_with_id = pd.concat([df["id"], one_hot_df], axis=1)

    # Save with ID included
    one_hot_df_with_id.to_csv(outputfile, index=False, encoding="utf-8")
    print("✅ One-hot encoded tags with ID saved to" +outputfile)
    return one_hot_df_with_id




# # Take the first 10 rows for testing
# df_test = df.head(10).copy()
#
# # --- Apply semantic tagging ---
# df_test["tags"] = df_test["description"].apply(tag_description)
#
# # --- Convert to one-hot encoding ---
# one_hot_test = df_test["tags"].str.get_dummies(sep=", ")
#
# # --- Display results ---
# print("Tagged descriptions:")
# print(df_test[["description", "tags"]])
# print("\nOne-hot encoded tags:")
# print(one_hot_test)
#
# # --- Optionally save to CSV for inspection ---
# df_test[["description", "tags"]].to_csv("test_tagged.csv", index=False, encoding="utf-8")
# one_hot_test.to_csv("test_one_hot.csv", index=False, encoding="utf-8")
