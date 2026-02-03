import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
from sklearn.utils import class_weight
import os
# 1. Setup Data Paths

TRAIN_DIR = 'train'
TEST_DIR = 'test'

# 2. Data Augmentation (Stronger to prevent overfitting)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 3. Load Images

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(48, 48), batch_size=32,
    color_mode="grayscale", class_mode='categorical', shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(48, 48), batch_size=32,
    color_mode="grayscale", class_mode='categorical', shuffle=False
)

# --- THE FIX: CALCULATE CLASS WEIGHTS ---

# This forces the model to care more about small folders like 'disgust'

labels = train_generator.classes
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights_dict = dict(enumerate(weights))
num_classes = len(train_generator.class_indices)

# 4. Improved Model Architecture

model = Sequential([
    Input(shape=(48, 48, 1)),

    # Block 1

    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 2

    Conv2D(128, (5,5), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 3

    Conv2D(512, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 5. Compile with a specific learning rate

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Callbacks: Slows down learning if accuracy stops improving

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# 7. Training

print(f"Starting training with Class Weights: {class_weights_dict}")
model.fit(
    train_generator,
    epochs=30, # Increased epochs to give it time to learn
    validation_data=test_generator,
    class_weight=class_weights_dict,
    callbacks=[reduce_lr]
)

# 8. Save

model.save('emotion_model.h5')
print("✅ New improved model saved as 'emotion_model.h5'")