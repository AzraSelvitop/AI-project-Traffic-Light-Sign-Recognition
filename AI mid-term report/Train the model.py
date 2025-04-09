import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json

# Paths to datasets
train_dir = 'C:/GTSRB/Train/Augmented_Images'
test_dir = 'C:/GTSRB/Test/Organized_Images'

# Image parameters
img_height, img_width = 64, 64
batch_size = 32

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Train and test generators
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
    shuffle=False
)

# Save class indices
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
print("Class indices saved to 'class_indices.json'.")

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(64, 64, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(128, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(2, 2),
    Dropout(0.4),

    Conv2D(256, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(2, 2),
    Dropout(0.4),

    Flatten(),
    Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
epochs = 3
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr]
)

# Save the trained model
model.save('traffic_sign_classifier.h5')
print("Model saved as 'traffic_sign_classifier.h5'.")
