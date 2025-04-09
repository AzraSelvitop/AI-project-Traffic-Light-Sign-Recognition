import os
import cv2
import random
from matplotlib import pyplot as plt

# Path to train dataset
train_dir = 'C:/TrafficLight/Processed_Dataset/Train'

# Function to resize and display one random image
def resize_one_random_image(input_dir, new_size=(224, 224)):
    # Pick a random class folder and image file
    class_folder = random.choice(os.listdir(input_dir))
    class_path = os.path.join(input_dir, class_folder)
    image_file = random.choice(os.listdir(class_path))
    image_path = os.path.join(class_path, image_file)

    # Read the image
    image = cv2.imread(image_path)
    if image is not None:
        print(f"Original image size: {image.shape[:2]}")
        
        # Resize the image
        resized_image = cv2.resize(image, new_size)
        print(f"Resized image size: {resized_image.shape[:2]}")

        # Display the image
        plt.figure(figsize=(8, 6))
        plt.title(f"Random Image from {class_folder}")
        plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        return resized_image
    else:
        print("Failed to load image. Please check the path.")
        return None

# Run the function
resize_one_random_image(train_dir)

