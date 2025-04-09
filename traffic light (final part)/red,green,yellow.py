import os
import cv2
import pandas as pd
from tqdm import tqdm

annotations_base_path = 'C:/TrafficLight/Annotations/dayTrain'
images_base_path = 'C:/TrafficLight/dayTrain'
processed_dataset_path = 'C:/TrafficLight/Processed_Dataset'

label_to_color = {
    'go': 'Green',
    'stop': 'Red',
    'warning': 'Yellow'
}

def extract_and_save_images():
    for clip_folder in os.listdir(annotations_base_path):
        annotations_file = os.path.join(annotations_base_path, clip_folder, 'frameAnnotationsBULB.csv')
        if not os.path.exists(annotations_file):
            print(f"Skipping {clip_folder}, no annotation file found.")
            continue

        annotations = pd.read_csv(annotations_file, sep=';')

        for index, row in tqdm(annotations.iterrows(), total=len(annotations), desc=f"Processing {clip_folder}"):
            label = row['Annotation tag'].strip().lower()
            if label not in label_to_color:
                continue  # Skip if it's not a recognized label

            color_label = label_to_color[label]
            image_path = os.path.join(images_base_path, clip_folder, os.path.basename(row['Filename']))

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue

            dest_folder = os.path.join(processed_dataset_path, 'Train', color_label)
            os.makedirs(dest_folder, exist_ok=True)

            dest_path = os.path.join(dest_folder, os.path.basename(image_path))
            cv2.imwrite(dest_path, image)

    print("Traffic light dataset extraction complete")

extract_and_save_images()
