import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to your datasets
train_dir = 'C:/GTSRB/Train/Augmented_Images'  # Augmented train dataset
test_dir = 'C:/GTSRB/Test/Organized_Images' # Test dataset with class folders

# Image parameters
img_height, img_width = 64, 64  # Resized image dimensions
batch_size = 32

# Data augmentation and normalization for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Normalization for test set (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load train dataset
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # Multi-class classification
)

# Load test dataset
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print("Train and Test datasets loaded successfully!")

