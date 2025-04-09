import os
import shutil
import random

def split_train_test(base_path, split_ratio=0.2):
    for color in ['Red', 'Yellow', 'Green']:
        source_folder = os.path.join(base_path, 'Train', color)
        dest_folder = os.path.join(base_path, 'Test', color)
        os.makedirs(dest_folder, exist_ok=True)

        # List all images in the source folder
        all_images = os.listdir(source_folder)
        num_test_images = int(len(all_images) * split_ratio)

        # Randomly select images for the test set
        test_images = random.sample(all_images, num_test_images)

        for img_name in test_images:
            src_path = os.path.join(source_folder, img_name)
            dest_path = os.path.join(dest_folder, img_name)
            shutil.move(src_path, dest_path)

    print("Train-Test split completed!")

# Run the split (80% training, 20% testing)
processed_dataset_path = 'C:/TrafficLight/Processed_Dataset'
split_train_test(processed_dataset_path)
