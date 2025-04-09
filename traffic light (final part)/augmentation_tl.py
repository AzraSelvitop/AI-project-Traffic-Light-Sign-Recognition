import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

# Paths
train_dir = 'C:/TrafficLight/Processed_Dataset/Train'
augmented_dir = 'C:/TrafficLight/Processed_Dataset/Augmented_Train'

# Augmentation configuration
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create augmented dataset folder
os.makedirs(augmented_dir, exist_ok=True)

for class_folder in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_folder)
    augmented_class_path = os.path.join(augmented_dir, class_folder)
    os.makedirs(augmented_class_path, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            img = img.reshape((1,) + img.shape)
            aug_iter = datagen.flow(img, batch_size=1)

            # Generate 5 augmented images per original image
            for i in range(5):
                aug_img = next(aug_iter)[0]
                aug_img_path = os.path.join(augmented_class_path, f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg")
                cv2.imwrite(aug_img_path, aug_img * 255)  # De-normalize before saving
print("Augmented images generated and saved.")
