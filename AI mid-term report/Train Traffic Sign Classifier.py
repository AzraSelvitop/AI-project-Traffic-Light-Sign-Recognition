import tensorflow as tf

# Paths to augmented dataset
train_dir = 'C:/GTSRB/Train/Augmented_Images'  # Path to class folders

# Image parameters
img_height, img_width = 64, 64
batch_size = 32

# ImageDataGenerator for loading images
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # For multi-class classification
)

print("Dataset loaded successfully!")
