import pandas as pd
import ImageAnalysis as ia
import time



# Load your CSV

start = time.time()
df = pd.read_csv("listings_ams.csv")

# Assume columns: 'id', 'image_url'
ids = df['id']
image_urls = df['picture_url']

results_list = []
i=0
for idx, url in zip(ids, image_urls):
    i+=1
    try:
        analysis = ia.analyze_image(url)
        analysis['id'] = idx
        analysis['url'] = url
        results_list.append(analysis)
    except Exception as e:
        print(f"Error processing id={idx}, url={url}: {e}")
        # You can append None or skip, here I skip
    if i==10:
        break

# Convert list of dicts to DataFrame
results_df = pd.DataFrame(results_list)

# Put 'id' as the first column for clarity
cols = ['id'] + [c for c in results_df.columns if c != 'id']
results_df = results_df[cols]

# Save to CSV
results_df.to_csv("analyzed_results2.csv", index=False)

print("Saved analysis results to analyzed_results2.csv")
end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")