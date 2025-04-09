from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to your dataset directories
train_dir = 'C:/GTSRB/Train/Augmented_Images'
test_dir = 'C:/GTSRB/Test/Organized_Images'

# Image parameters
img_height, img_width = 64, 64
batch_size = 32

# Define ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators for training and testing datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Keep testing data in order
)

# Step 1: Print class indices
print("Training Class Indices:", train_generator.class_indices)
print("Testing Class Indices:", test_generator.class_indices)

# Step 2: Compare class indices
if train_generator.class_indices == test_generator.class_indices:
    print("\nClass mappings are consistent between training and testing datasets.")
else:
    print("\nClass mappings are inconsistent! Verify folder names and structure.")

# Step 3: Check example labels from training data
x_train, y_train = next(train_generator)
print("\nExample Training Labels (One-Hot):")
for i, label in enumerate(y_train[:5]):
    print(f"Image {i+1}: {label}")

# Step 4: Check example labels from testing data
x_test, y_test = next(test_generator)
print("\nExample Testing Labels (One-Hot):")
for i, label in enumerate(y_test[:5]):
    print(f"Image {i+1}: {label}")
