import tensorflow as tf
import numpy as np
import cv2
import os
import json

model = tf.keras.models.load_model('traffic_sign_classifier.h5')
print("Model loaded successfully.")

with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

class_labels = {v: k for k, v in class_indices.items()}

test_images_dir = 'C:/GTSRB/testing_images'  

sign_names = {
    "00000": "Speed Limit (20km/h)",
    "00001": "Speed Limit (30km/h)",
    "00002": "Speed Limit (50km/h)",
    "00003": "Speed Limit (60km/h)",
    "00004": "Speed Limit (70km/h)",
    "00005": "Speed Limit (80km/h)",
    "00006": "Speed Limit (90km/h)",
    "00007": "Speed Limit (100km/h)",
    "00008": "Speed Limit (120km/h)",
    "00009": "Speed Limit (50km/h)",
    "00010": "Speed Limit (50km/h)",
    "00011": "Speed Limit (50km/h)",
    "00012": "Speed Limit (50km/h)",
    "00013": "Falling Rocks",
    "00014": "Stop",
    "00015": "No parking",
    "00016": "Speed Limit (50km/h)",
    "00017": "Speed Limit (50km/h)",
    "00018": "Caution",
    "00019": "Right hand curve ",
    "00020": "Left hand curve",
    "00021": "Right reverse bend",
    "00022": "Hump or rough road",
    "00023": "Slippery road",
    "00024": "Lane ends",
    "00025": "Road work ahead",
    "00026": "Traffic signals ahead",
    "00027": "Pedestrian crossing ahead",
    "00028": "Children",
    "00029": "Bicycle",
    "00030": "Snow",
    "00031": "Deer",
    "00032": "End of all bans",
    "00033": "Turn right",
    "00034": "Turn left",
    "00035": "Go straight",
    "00036": "Straight and right turn",
    "00037": "Straight and left turn",
    "00038": "Children",
    "00039": "Children",
    "00040": "Circular arrow",
    "00041": "Circular arrow",
    "00042": "Circular arrow",

}

for img_name in os.listdir(test_images_dir):
    img_path = os.path.join(test_images_dir, img_name)

   
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_name}")
        continue

    resized_img = cv2.resize(img, (64, 64)) 
    input_img = np.expand_dims(resized_img / 255.0, axis=0)  

    prediction = model.predict(input_img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    class_id = class_labels[predicted_class]
    sign_name = sign_names.get(class_id, "Unknown Sign")

    text = f"Sign: {sign_name} | Confidence: {confidence:.2f}%"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Black text
    cv2.imshow("Traffic Sign Prediction", img)
    cv2.waitKey(0) 

cv2.destroyAllWindows()
