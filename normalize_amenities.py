import pandas as pd
import re
import ast
from tqdm import tqdm
from collections import defaultdict
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os  # Import the os module to handle file paths

# --- Library Imports for Flair ---
try:
    from flair.data import Sentence
    from flair.models import SequenceTagger

    FLAIR_AVAILABLE = True
except ImportError:
    print("Warning: Flair library not found. Brand removal will be skipped.")
    FLAIR_AVAILABLE = False

# --- NLP Setup ---
try:
    NLP = spacy.load("en_core_web_md")
    print("✓ spaCy NLP model (md) with word vectors loaded successfully.")
except OSError:
    print("Error: spaCy model 'en_core_web_md' not found. Please run: python -m spacy download en_core_web_md")
    exit()

# --- Setup: Define Stop Words ---
CUSTOM_STOP_WORDS = set(CountVectorizer(stop_words='english').get_stop_words()).union(
    ['listing', 'extra', 'cost', 'included', 'private'])


# --- Cleaning Function ---
def clean_amenity_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in CUSTOM_STOP_WORDS]
    text = ' '.join(filtered_words)
    return re.sub(r'\s+', ' ', text).strip()


# --- Flair Brand Remover Class ---
class FlairBrandRemover:
    def __init__(self):
        self.tagger = None
        if FLAIR_AVAILABLE:
            try:
                self.tagger = SequenceTagger.load('ner-large')
                print("✓ Flair NER model loaded successfully.")
            except Exception as e:
                print(f"Warning: Could not load Flair NER model. Error: {e}")
                self.tagger = None

    def remove_brands(self, text):
        if not self.tagger or not text: return text
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        brands_to_remove = {entity.text.lower() for entity in sentence.get_spans('ner') if
                            entity.tag in ['ORG', 'PER', 'MISC']}
        if not brands_to_remove: return text
        words = text.split()
        non_brand_words = [word for word in words if word.lower() not in brands_to_remove]
        return ' '.join(non_brand_words)


# --- Semantic Classifier Class ---
class AmenityClassifier:
    def __init__(self, category_definitions):
        self.category_vectors = {}
        print("Calculating semantic vectors for categories...")
        for category, examples in tqdm(category_definitions.items(), desc="Vectorizing Categories"):
            example_vectors = [NLP(example).vector for example in examples if NLP(example).has_vector]
            if example_vectors:
                self.category_vectors[category] = np.mean(example_vectors, axis=0)

    def _cosine_similarity(self, vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0: return 0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def predict(self, amenity_text, threshold=0.4):
        if not amenity_text or not NLP(amenity_text).has_vector: return None
        amenity_vector = NLP(amenity_text).vector
        best_category, max_similarity = None, -1
        for category, category_vector in self.category_vectors.items():
            similarity = self._cosine_similarity(amenity_vector, category_vector)
            if similarity > max_similarity:
                max_similarity, best_category = similarity, category
        if max_similarity > threshold:
            return best_category
        return None


# --- Main Logic ---
AMENITY_CATEGORIES = {
    'safety_core': ['smoke alarm', 'carbon monoxide alarm', 'fire extinguisher', 'lock'],
    'safety_extra': ['first aid', 'lockbox', 'keypad', 'safe', 'guards'],
    'kitchen_core': ['kitchen', 'refrigerator', 'fridge', 'oven', 'stove', 'cooktop', 'microwave', 'kettle',
                     'hot water kettle', 'freezer'],
    'kitchen_extra': ['dishwasher', 'blender', 'baking', 'toaster', 'kitchenette', 'rice maker'],
    'laundry_core': ['washer', 'dryer', 'washing machine', 'tumble dryer', 'iron', 'drying rack'],
    'laundry_extra': ['laundromat'],
    'air_conditioner_core': ['air conditioning', 'ac', 'aircon'],
    'air_conditioner_extra': ['fan'],
    'heater_core': ['heating'],
    'heater_extra': ['heater', 'fireplace'],
    'hygiene_core': ['shampoo', 'soap', 'towel', 'shower', 'hot water', 'essentials', 'conditioner', 'body soap'],
    'hygiene_extra': ['bathtub', 'bath', 'bidet'],
    'wellness_core': ['sauna', 'hot tub', 'jacuzzi', 'spa'],
    'storage_core': ['closet', 'wardrobe', 'dresser'],
    'storage_extra': ['hangers'],
    'entertainment_core': ['tv', 'hdtv', 'television'],
    'entertainment_extra': ['hbo', 'cable', 'netflix', 'hulu', 'amazon prime', 'disney', 'books', 'games', 'console',
                            'record player', 'piano'],
    'internet_core': ['wifi', 'internet', 'wireless internet', 'ethernet'],
    'parking_core': ['parking', 'garage', 'carport'],
    'parking_extra': ['ev charger'],
    'exercise_core': ['gym', 'fitness', 'exercise', 'weights', 'treadmill', 'yoga mat', 'ping pong table'],
    'exercise_extra': ['climbing wall'],
    'outdoors_core': ['patio', 'balcony', 'pool'],
    'outdoors_extra': ['grill', 'bbq', 'lounger', 'lounge', 'fire pit', 'firepit', 'sun lounger'],
    'children_care_core': ['crib', 'high chair', 'children', 'baby', 'changing table'],
    'children_care_extra': ['childrens dinnerware', 'babysitter'],
    'cleaning_core': ['cleaning'],
    'cleaning_extra': ['housekeeping'],
    'pets_allowed': ['pets allowed'],
    'smoking_allowed': ['smoking allowed'],
    'coffee_maker_core': ['coffee maker', 'coffee', 'nespresso', 'keurig'],
    'cooking_basics_core': ['cooking basics', 'dinnerware', 'table'],
    'sound_system_core': ['sound system', 'sound', 'speakers'],
    'accessibility_core': ['elevator', 'lift', 'wheelchair ramp', 'ground floor'],
    'view_core': ['view', 'waterfront'],
    'bedroom_core': ['bed', 'pillow', 'linens', 'blanket', 'duvet', 'sheet'],
    'bedroom_extra': ['room darkening shades'],
    'attractions_nearby_core': ['restaurant', 'coffee shop', 'lake', 'beach', 'river', 'center', 'golf', 'bowling',
                                'resort', 'ski'],
    'attractions_nearby_extra': ['museum'],
    'hospitality_core': ['self check-in'],
    'hospitality_extra': ['host greets', 'building staff', 'breakfast'],
    'transport_core': ['bus stop', 'bike', 'boat', 'bicycle', 'train'],
    'transport_extra': ['kayak']
}

CATEGORY_REPRESENTATIVES = {
    'exercise_core': ['gym', 'exercise equipment', 'weights', 'treadmill', 'yoga mat', 'ping pong table'],
    'kitchen_core': ['kitchen', 'appliance', 'refrigerator', 'oven', 'stove', 'fridge'],
    'safety_core': ['safety', 'security', 'alarm', 'detector', 'lock', 'first aid'],
    'outdoors_core': ['outdoors', 'backyard', 'garden', 'patio', 'balcony', 'grill', 'bbq'],
    'cooking_basics_core': ['cooking basics', 'dinnerware', 'utensils', 'cups', 'glasses', 'forks', 'spoons', 'pots']
}


def classify_single_amenity(raw_amenity, category_map, classifier, brand_remover):
    raw_amenity_lower = raw_amenity.lower()
    for category, search_terms in category_map.items():
        for term in search_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', raw_amenity_lower):
                return category
    brand_free_text = brand_remover.remove_brands(raw_amenity)
    cleaned_text = clean_amenity_text(brand_free_text)
    return classifier.predict(cleaned_text)


def inspect_amenity_classification(df, classifier, brand_remover):
    all_unique_amenities = set(a for amenities_list in df['parsed_amenities'] for a in amenities_list)
    print(f"\nFound {len(all_unique_amenities)} unique amenities to classify.")
    final_groups = defaultdict(list)
    for raw_amenity in tqdm(sorted(list(all_unique_amenities)), desc="Classifying Amenities"):
        category = classify_single_amenity(raw_amenity, AMENITY_CATEGORIES, classifier, brand_remover)
        if category:
            final_groups[category].append(raw_amenity)
        else:
            final_groups['__UNCLASSIFIED__'].append(raw_amenity)
    return final_groups


def create_binary_columns(df, final_groups):
    print("\nCreating binary columns...")
    amenity_to_category_map = {}
    for category, amenities in final_groups.items():
        if category == '__UNCLASSIFIED__': continue
        for amenity in amenities:
            amenity_to_category_map[amenity] = category
    all_categories = sorted([cat for cat in final_groups.keys() if cat != '__UNCLASSIFIED__'])

    tqdm.pandas(desc="Mapping Amenities to Categories")
    df['amenity_categories'] = df['parsed_amenities'].progress_apply(
        lambda amenities_list: {amenity_to_category_map.get(a) for a in amenities_list if
                                amenity_to_category_map.get(a)}
    )
    for category in tqdm(all_categories, desc="Creating Binary Columns"):
        column_name = f"has_{category}"
        df[column_name] = df['amenity_categories'].apply(lambda cat_set: 1 if category in cat_set else 0)
    df = df.drop(columns=['parsed_amenities', 'amenity_categories'])
    print("✓ Binary columns created successfully.")
    return df

def normalize_amenities(df,output_file='cleaned_norm_amenities.csv'):
    print("\nNormalizing Amenities...")
    # --- Initialization ---
    classifier = AmenityClassifier(CATEGORY_REPRESENTATIVES)
    brand_remover = FlairBrandRemover()


    # df = pd.read_csv('clean_merged_database.csv', engine='python', on_bad_lines='warn')
    tqdm.pandas(desc="Parsing Amenities")
    df['parsed_amenities'] = df['amenities'].progress_apply(
        lambda s: ast.literal_eval(s.strip()) if isinstance(s, str) and s.strip().startswith('[') else []
    )

    # --- Step 1: Inspect the classification ---
    final_classified_groups = inspect_amenity_classification(df.copy(), classifier, brand_remover)

    print("\n--- Amenity Classification Review ---")
    for category, amenities in sorted(final_classified_groups.items()):
        print(f"\n--- Category: {category} ({len(amenities)} items) ---")
        for amenity in amenities[:100]:
            print(f"  - {amenity}")
        if len(amenities) > 100:
            print(f"  - ... and {len(amenities) - 100} more.")

    # --- Step 2: Create the final DataFrame with binary columns ---
    transformed_df = create_binary_columns(df, final_classified_groups)

    # --- Step 3: Display the final results ---
    print("\nTransformation complete. Here's a preview of the new columns:")
    amenity_cols = sorted([col for col in transformed_df.columns if col.startswith('has_')])
    display_cols = ['id', 'name'] + amenity_cols
    if len(display_cols) > 20:
        print(f"(Showing a subset of the {len(amenity_cols)} new amenity columns)")
        display_cols = display_cols[:20]
    print(transformed_df[display_cols].head())

    # --- Step 4: Save the new DataFrame to a CSV file ---
    # Create the output filename dynamically
    # base_name = os.path.basename(csv_file_path)
    # name, ext = os.path.splitext(base_name)
    # output_filename = f"{name}_with_amenity_cols{ext}"
    result_cols = ['id',
                   'has_accessibility_core',
                   'has_air_conditioner_core',
                   'has_air_conditioner_extra',
                   'has_attractions_nearby_core',
                   'has_bedroom_core',
                   'has_children_care_core',
                   'has_children_care_extra',
                   'has_cleaning_core',
                   'has_cleaning_extra',
                   'has_coffee_maker_core',
                   'has_cooking_basics_core',
                   'has_entertainment_core',
                   'has_entertainment_extra',
                   'has_exercise_core',
                   'has_exercise_extra',
                   'has_heater_core',
                   'has_heater_extra',
                   'has_hospitality_core',
                   'has_hospitality_extra',
                   'has_hygiene_core',
                   'has_hygiene_extra',
                   'has_internet_core',
                   'has_kitchen_core',
                   'has_kitchen_extra',
                   'has_laundry_core',
                   'has_laundry_extra',
                   'has_outdoors_core',
                   'has_outdoors_extra',
                   'has_parking_core',
                   'has_parking_extra',
                   'has_pets_allowed',
                   'has_safety_core',
                   'has_safety_extra',
                   'has_smoking_allowed',
                   'has_sound_system_core',
                   'has_storage_core',
                   'has_storage_extra',
                   'has_transport_core',
                   'has_transport_extra',
                   'has_view_core',
                   'has_wellness_core']

    print(f"\nSaving transformed DataFrame to: {output_file}")
    # Use index=False to avoid writing the DataFrame index as a column
    transformed_df[result_cols].to_csv(output_file, index=False)
    print("✓ File saved successfully.")
# --- Main execution block ---
if __name__ == '__main__':
    df=pd.read_csv('clean_merged_database.csv')
    normalize_amenities(df,'normalized_amenities.csv')



