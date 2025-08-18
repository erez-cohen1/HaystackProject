import pandas as pd
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
        results_df.to_csv("image_analysis.csv", mode='w', index=False)

        end = time.time()
        print(f"Elapsed time: {end - start:.4f} seconds")
    # Save failed IDs
    if failed_images:
        with open("failed_images.txt", "w") as f:
            f.write("\n".join(map(str, failed_images)))
    print(f"Processed {len(all_results)} images. Failed: {len(failed_images)}")



if __name__ == '__main__':
    df=pd.read_csv("listings_clean.csv")
    create_image_db(df)
