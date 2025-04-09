import cv2
import os


resize_width, resize_height = 224, 224

dataset_path = 'C:/TrafficLight/Processed_Dataset' 


for folder in ['Train', 'Test']:
    folder_path = os.path.join(dataset_path, folder)
    if not os.path.exists(folder_path): 
        print(f"Folder not found: {folder_path}")
        continue

    for class_folder in ['Red', 'Yellow', 'Green']:
        class_path = os.path.join(folder_path, class_folder)
        if not os.path.exists(class_path):  
            print(f"Class folder not found: {class_path}")
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:  
                    print(f"Failed to load image: {img_path}")
                    continue

         
                resized_img = cv2.resize(img, (resize_width, resize_height))
                cv2.imwrite(img_path, resized_img) 

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

print("Image resizing completed successfully!")
