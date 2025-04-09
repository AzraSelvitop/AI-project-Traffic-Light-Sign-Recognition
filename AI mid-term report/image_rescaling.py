import os
import cv2

# For Train folder (images in subdirectories)
def resize_train_images(input_folder, output_folder, size=(64, 64)):
    os.makedirs(output_folder, exist_ok=True)

    for root, _, files in os.walk(input_folder):  # Recursive traversal for subdirectories
        for file in files:
            if file.lower().endswith(('.ppm', '.jpg', '.png')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Could not read file: {img_path}")
                    continue
                
                resized = cv2.resize(img, size)
                
                # Maintain folder structure
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)
                
                # Save the resized image
                output_path = os.path.join(output_subfolder, file)
                cv2.imwrite(output_path, resized)
                print(f"Resized and saved: {output_path}")

# For Test folder (images directly in one folder)
def resize_test_images(input_folder, output_folder, size=(64, 64)):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):  # Non-recursive traversal for flat folder
        if file.lower().endswith(('.ppm', '.jpg', '.png')):
            img_path = os.path.join(input_folder, file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Could not read file: {img_path}")
                continue
            
            resized = cv2.resize(img, size)
            output_path = os.path.join(output_folder, file)
            cv2.imwrite(output_path, resized)
            print(f"Resized and saved: {output_path}")

# Example usage
resize_train_images('C:/GTSRB/Train/Images/GTSRB/Final_Training/Images', 'C:/GTSRB/Train/resized_images')
resize_test_images('C:/GTSRB/Test/Images/GTSRB/Final_Test/Images', 'C:/GTSRB/Test/resized_images')


