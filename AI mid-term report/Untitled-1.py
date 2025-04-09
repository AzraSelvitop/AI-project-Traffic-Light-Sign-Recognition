import os

# Define the paths (update these paths to match your file structure)
test_images_path = 'C:/GTSRB/Test/Images'
haar_path = 'C:/GTSRB/Test/HaarFeatures'
hog_path = 'C:/GTSRB/Test/HOGFeatures'
hue_path = 'C:/GTSRB/Test/HueHistogram'

# Verify all paths exist
for path in [test_images_path, haar_path, hog_path, hue_path]:
    if not os.path.exists(path):
        print(f"Directory not found: {path}")
        exit(1)

# Load file sets
test_images = set(os.listdir(test_images_path))

haar = set(os.listdir(haar_path))
hog = set(os.listdir(hog_path))
hue = set(os.listdir(hue_path))

# Check for missing features
for img in test_images:
    base_name = os.path.splitext(img)[0]  # Get filename without extension


    if not any(base_name in f for f in haar):
        print(f"Missing Haar feature for {img}")
    if not any(base_name in f for f in hog):
        print(f"Missing HOG feature for {img}")
    if not any(base_name in f for f in hue):
        print(f"Missing HueHistogram for {img}")
