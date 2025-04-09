import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model('best_traffic_light_model.keras')
print("Model loaded successfully.")

# Define the classes and actions
classes = ['Red', 'Yellow', 'Green']
actions = {
    'Red': 'Stop',
    'Yellow': 'Warning',
    'Green': 'Go'
}

# Directory containing test images
test_images_dir = 'C:/TrafficLight/deneme resimleri'  # Replace with your folder path

# Loop through images in the folder
for img_name in os.listdir(test_images_dir):
    img_path = os.path.join(test_images_dir, img_name)

    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_name}")
        continue

    # Resize and normalize the image for the model
    resized_img = cv2.resize(img, (224, 224))  # Match model input size
    input_img = np.expand_dims(resized_img / 255.0, axis=0)  # Normalize and add batch dimension

    # Predict the traffic light state
    prediction = model.predict(input_img)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display the result
    print(f"Image: {img_name}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    # Annotate the image with predictions
    cv2.putText(img, f"Class: {predicted_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black text
    cv2.putText(img, f"Confidence: {confidence:.2f}%", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black text

    # Display the image
    cv2.imshow("Traffic Light Prediction", img)
    cv2.waitKey(0)  # Press any key to move to the next image

cv2.destroyAllWindows()
