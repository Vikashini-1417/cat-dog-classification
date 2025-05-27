# STEP 1: UNZIP THE DATASET
import zipfile
import os

zip_path = "archive (11).zip"  # your uploaded ZIP
extract_path = "/content/dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Check folders
print("Extracted folders:", os.listdir(extract_path))

# STEP 2: SETUP DATA GENERATORS
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = os.path.join(extract_path, 'train')  # Adjust if folder name differs
test_dir = os.path.join(extract_path, 'test')

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# STEP 3: BUILD THE CNN MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# STEP 4: COMPILE AND TRAIN THE MODEL
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# STEP 5: EVALUATE THE MODEL
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc * 100:.2f}%")

# STEP 6: UPLOAD AND TEST A NEW IMAGE
from google.colab import files
uploaded = files.upload()  # Upload a single cat/dog image

# STEP 7: PREDICT THE IMAGE CLASS
import numpy as np
from tensorflow.keras.preprocessing import image

filename = next(iter(uploaded))  # get the uploaded filename

# Load and preprocess the image
img_path = filename
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)

# Output
if prediction[0] < 0.5:
    print("Prediction: Cat 🐱")
else:
    print("Prediction: Dog 🐶")