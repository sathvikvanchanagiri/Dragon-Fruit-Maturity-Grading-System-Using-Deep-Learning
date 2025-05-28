# Example code to download and extract images from Open Images Dataset
import os
from bing_image_downloader import downloader

# Define the path to the "Other" directory
dataset_path = "Dragon Fruit Maturity Detection Dataset/Augmented Dataset/Other"
os.makedirs(dataset_path, exist_ok=True)

def download_bing_images(num_images=200):
    # Define search queries for diverse "Other" images
    queries = [
        "random objects",
        "nature scenes",
        "animals",
        "food items",
        "household items"
    ]
    
    # Download images for each query
    for query in queries:
        try:
            print(f"Downloading images for query: {query}")
            # Temporary directory for download
            temp_dir = os.path.join(dataset_path, "temp")
            
            # Download to temp directory
            downloader.download(
                query,
                limit=int(num_images / len(queries)),
                output_dir=temp_dir,
                adult_filter_off=True,
                force_replace=False,
                timeout=60,
                verbose=True
            )
            
            # Move files from temp directory to main Other directory
            query_dir = os.path.join(temp_dir, query)
            if os.path.exists(query_dir):
                for filename in os.listdir(query_dir):
                    src = os.path.join(query_dir, filename)
                    dst = os.path.join(dataset_path, filename)
                    os.rename(src, dst)
                
                # Remove empty temp directory
                os.rmdir(query_dir)
                os.rmdir(temp_dir)
                
        except Exception as e:
            print(f"Error downloading images for query {query}: {str(e)}")

# Download images for the "Other" class
print(f"Creating 'Other' dataset in: {dataset_path}")
download_bing_images(200)
print("Dataset creation complete!")