import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2)
)

# Paths
input_dir = 'C:/GTSRB/Train/Resized_Images'
output_dir = 'C:/GTSRB/Train/Augmented_Images'
os.makedirs(output_dir, exist_ok=True)

# Traverse subdirectories
for root, _, files in os.walk(input_dir):  # Recursively go through subfolders
    for file in files:
        if file.lower().endswith(('.ppm', '.jpg', '.png')):  # Check valid image formats
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error reading: {img_path}")
                continue

            print(f"Processing: {img_path}")

            # Reshape and apply augmentation
            img = cv2.resize(img, (64, 64))  # Resize to consistent dimensions
            x = img.reshape((1,) + img.shape)  # Reshape for generator

            # Generate augmented images
            subfolder = os.path.relpath(root, input_dir)  # Get subfolder structure
            output_subdir = os.path.join(output_dir, subfolder)
            os.makedirs(output_subdir, exist_ok=True)  # Ensure subfolder exists

            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_subdir, save_prefix='aug', save_format='jpeg'):
                i += 1
                print(f"Generated augmented image for {file} in {output_subdir}")
                if i > 5:  # Limit to 5 augmented images per original
                    break
