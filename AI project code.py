import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# === Step 1: Load Data ===
def load_data(data_dir):
    images = []
    labels = []
    for label_folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, label_folder)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, file_name)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (64, 64))  # Resize to 64x64
                images.append(image)
                labels.append(int(label_folder))  # Folder name as label
    return np.array(images), np.array(labels)

train_dir = r"C:\AI n CV\datasheet\train"
test_dir = r"C:\AI n CV\datasheet\test"


print("Loading dataset...")
train_images, train_labels = load_data(train_dir)
test_images, test_labels = load_data(test_dir)
print("Dataset loaded successfully!")

# === Step 2: Preprocess Images ===
def preprocess_images(images):
    # Convert to grayscale
    gray_images = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images])
    # Normalize pixel values
    normalized_images = gray_images / 255.0
    return normalized_images

print("Preprocessing images...")
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

# === Step 3: Feature Extraction (HOG) ===
def extract_hog_features(images):
    hog = cv2.HOGDescriptor()
    features = np.array([hog.compute(img).flatten() for img in images])
    return features

print("Extracting HOG features...")
train_features = extract_hog_features(train_images)
test_features = extract_hog_features(test_images)

# === Step 4: Train-Test Split (Optional) ===
# Split training data into train and validation sets
train_features, val_features, train_labels, val_labels = train_test_split(
    train_features, train_labels, test_size=0.2, random_state=42
)

# === Step 5: Train SVM Classifier ===
print("Training SVM classifier...")
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

svm = SVC(kernel='linear', probability=True)
svm.fit(train_features, train_labels)

# Validate the model
val_predictions = svm.predict(val_features)
val_accuracy = accuracy_score(val_labels, val_predictions)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# === Step 6: Test the Model ===
print("Testing the model...")
test_predictions = svm.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# === Step 7: Visualize Results ===
def visualize_predictions(images, predictions, labels, num_samples=5):
    for i in range(num_samples):
        img = images[i]
        pred = predictions[i]
        true_label = labels[i]
        display = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(display, f"Pred: {pred}, True: {true_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Result", display)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Visualizing some predictions...")
visualize_predictions(test_images, test_predictions, test_labels)

# === Step 8: Save Model ===
print("Saving the trained model...")
with open("svm_model.pkl", "wb") as f:
    pickle.dump((svm, scaler), f)
print("Model saved successfully!")
