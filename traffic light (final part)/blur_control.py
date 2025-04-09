import os
import cv2
import numpy as np

def check_image_quality(folder_path, threshold=10):
    """
    Detects low-quality or blurry images in the specified folder.
    :param folder_path: Path to the folder containing images.
    :param threshold: Blurriness threshold (lower means blurrier images).
    :return: List of low-quality image filenames.
    """
    low_quality_images = []

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # Compute Laplacian variance (blurriness measure)
            variance = cv2.Laplacian(img, cv2.CV_64F).var()
            if variance < threshold:
                low_quality_images.append(img_name)
        else:
            print(f"Unable to read: {img_name}")

    return low_quality_images

# Path to the red folder
red_folder_path = "C:/TrafficLight/Processed_Dataset/Train/Red"

# Check for low-quality images
low_quality = check_image_quality(red_folder_path)

print(f"Low-quality images found: {len(low_quality)}")
print("Images:", low_quality)
