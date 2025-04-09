import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# Load the trained model
model = load_model('traffic_light_classifier.h5')

# Define the correct input size (same as used during training)
img_height, img_width = 64, 64  # Change according to your training size

# Load the test dataset
test_dir = 'C:/TrafficLight/Processed_Dataset/Test'
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Get a batch of test images
test_images, test_labels = next(test_generator)

# Resize the test images to match the input shape of the model
resized_images = np.array([cv2.resize(img, (img_height, img_width)) for img in test_images])

# Make predictions
predictions = model.predict(resized_images)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Class labels mapping
class_names = list(test_generator.class_indices.keys())

# Visualize some predictions
plt.figure(figsize=(10, 10))
for i in range(9):  # Show 9 random images
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i])
    true_label = class_names[true_labels[i]]
    predicted_label = class_names[predicted_labels[i]]
    plt.title(f"True: {true_label}\nPred: {predicted_label}")
    plt.axis('off')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_names))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))
