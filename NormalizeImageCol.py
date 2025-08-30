import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

import ImageAnalysis as ia
import time



def create_image_db(df, batch_size=8):
    """Process images in batches of 8 and save results"""
    all_results = []
    failed_images = []
    start = time.time()
    # Process in batches
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        image_urls = batch['picture_url'].tolist()

        # Get analysis for this batch
        result_batch = ia.analyze_image_batch(image_urls)  # returns List of dicts

        # Process each image's results
        for idx, url, result in zip(batch['id'], batch['picture_url'], result_batch):
            if result is None:
                failed_images.append(idx)
                continue

            # Add both ID and URL
            result['id'] = idx
            result['image_url'] = url  # <-- New line
            all_results.append(result)
        print(f"Batch #{i}: {len(all_results)} finished")

    # Convert to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        cols = ['id'] + [c for c in results_df.columns if c != 'id']
        results_df = results_df[cols]

        # Save to CSV (append if file exists)
        results_df.to_csv("image_analysis_merged.csv", mode='w', index=False)

        end = time.time()
        print(f"Elapsed time: {end - start:.4f} seconds")
    # Save failed IDs
    if failed_images:
        with open("failed_images.txt", "w") as f:
            f.write("\n".join(map(str, failed_images)))
    print(f"Processed {len(all_results)} images. Failed: {len(failed_images)}")



def cluster_thresholds(feature_series, n_bins=3):
    """Returns optimal thresholds using K-means clustering"""
    # Reshape data for clustering
    X = feature_series.dropna().values.reshape(-1, 1)

    # Fit K-means
    kmeans = KMeans(n_clusters=n_bins, random_state=42).fit(X)
    centers = sorted(kmeans.cluster_centers_.flatten())

    # Calculate midpoints between centers as thresholds
    thresholds = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]

    return [feature_series.min()] + thresholds + [feature_series.max()]

def normalize_img_cols(df):

    #remove images below 0.7 confidence
    df = df[df['room_confidence'] >=0.7].copy()

    #resolution


    #brightness
    col_brightness='brightness'
    # calculate_percentile_bins(df,col_brightness)
    bins= [(0,115.1),(115.2,130.1),(130.2,151.3),(151.4,None)]
    bright_labels = ['bright_low', 'bright_medium', 'bright_high', 'bright_very_high']
    df=bucketize_column(df,col_brightness,bins,bright_labels)


    #mood to 2 cols warm and cool
    df['warm'] = (df['mood'] == 'warm').astype(int)
    df['cool'] = (df['mood'] == 'cool').astype(int)

    #room types
    room_labels = ["bedroom", "kitchen", "bathroom", "living room", "dining room", "balcony","not a room",
                   "exterior of house","view from window","boat"]
    # Create a mapping dictionary
    room_mapping = {
        "not a room": "not a room",
        "exterior of house": "not a room",
        "view from window": "not a room",
        "boat": "not a room"
    }

    # Apply the mapping to your DataFrame
    df['room_type_consolidated'] = df['room_type'].replace(room_mapping)

    # Get unique room types after consolidation
    unique_rooms = ["bedroom", "kitchen", "bathroom", "living room", "dining room", "balcony", "not a room"]

    # Create binary columns
    for room in unique_rooms:
        df[room] = (df['room_type_consolidated'] == room).astype(int)

    #resolution
    df['resolution'] = df['resolution'].str.extract(r'(\d+),\s*(\d+)').apply(lambda x: (int(x[0]), int(x[1])), axis=1)
    df['res_bin'] = df['resolution'].apply(classify_resolution)
    df['low_res'] = (df['res_bin'] == 'low_res').astype(int)
    df['mid_res'] = (df['res_bin'] == 'mid_res').astype(int)
    df['high_res'] = (df['res_bin'] == 'high_res').astype(int)

    result_cols=(['id','warm','cool','low_res','mid_res','high_res'
                  ] + bright_labels +unique_rooms
                 )
    result = df[result_cols]
    result.columns = [
        f'img_{col.replace(" ", "_")}' if col != 'id' else col
        for col in result.columns
    ]
    result.to_csv('image_analysis_norm.csv', index=False)
    return result

def classify_resolution(res_tuple):

    megapixels = (res_tuple[0] * res_tuple[1]) / 1e6
    if megapixels < 2: return 'low_res'
    elif megapixels < 8: return 'mid_res'
    else: return 'high_res'


def calculate_percentile_bins(df,col):
    bins = np.percentile(df[col].dropna(), [25, 50, 75])
    df['temp_bin'] = pd.cut(
        df[col],
        bins=[-np.inf] + bins.tolist() + [np.inf],
        labels=['low', 'medium', 'high', 'very_high']
    )

    print(f"Thresholds: {bins}")
    print(df['temp_bin'].value_counts())

def plot_clusters(feature_series, thresholds):
    plt.figure(figsize=(10, 4))
    # Histogram
    plt.hist(feature_series, bins=50, alpha=0.5, label='Distribution')
    # Threshold lines
    for thresh in thresholds[1:-1]:
        plt.axvline(thresh, color='red', linestyle='--', label='Threshold')
    plt.title(f"{feature_series.name} Thresholds")
    plt.legend()
    plt.show()

def bucketize_column(df, col, bins, labels=None):
    """
    Create binary columns for ranges in a numeric column.
    make sure bins dont overlap.
    Parameters:
        df (pd.DataFrame): The DataFrame.
        col (str): Column to bucketize.
        bins (list of tuples): Each tuple is (min_val, max_val), inclusive. Use None for open-ended.
        labels (list of str, optional): Names for the new columns. If None, auto-generated.

    Returns:
        pd.DataFrame: DataFrame with new binary columns added.
    """
    if labels is None:
        labels = [f"{col}_{i}" for i in range(len(bins))]

    for (min_val, max_val), label in zip(bins, labels):
        if min_val is None:
            df[label] = (df[col] <= max_val).astype(int)
        elif max_val is None:
            df[label] = (df[col] >= min_val).astype(int)
        else:
            df[label] = ((df[col] >= min_val) & (df[col] <= max_val)).astype(int)
    return df
def create_normalize_image_db(df):
    create_image_db(df,64)
    df = pd.read_csv("image_analysis_merged.csv")
    return normalize_img_cols(df)

if __name__ == '__main__':

    # # df=pd.read_csv("merged_database.csv")
    # # create_image_db(df,64)    # df = pd.read_csv('image_analysis.csv')
    df=pd.read_csv("image_analysis_merged.csv")
    normalize_img_cols(df)

