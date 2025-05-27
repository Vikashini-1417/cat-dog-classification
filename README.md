# 🐱🐶 Cat vs Dog Image Classification using CNN

This project builds a Convolutional Neural Network (CNN) in TensorFlow/Keras to classify images as **cat** or **dog**. It uses an image dataset stored in a ZIP archive and provides functionality to train the model, evaluate accuracy, and test predictions on new images.

---

## 📁 Dataset Structure

Make sure your extracted dataset folder follows this format:

dataset/
├── train/
│ ├── cats/
│ └── dogs/
└── test/
├── cats/
└── dogs/

yaml
Copy
Edit

- `train/`: Training images (cats and dogs)
- `test/`: Testing images (cats and dogs)

---

## 🚀 Steps in the Project

### 1. **Unzip the Dataset**

```python
import zipfile
zip_path = "archive (11).zip"
extract_path = "/content/dataset"
The dataset ZIP file is extracted to the specified folder.

2. Setup Image Generators

ImageDataGenerator(rescale=1./255)
Used to load and normalize image pixels for training and testing.

3. Build CNN Model
The CNN consists of:

Three convolutional layers with max pooling

One dense hidden layer with dropout

A final sigmoid layer for binary classification

model = Sequential([
    Conv2D(32, ...), MaxPooling2D(...),
    Conv2D(64, ...), MaxPooling2D(...),
    Conv2D(128, ...), MaxPooling2D(...),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
4. Compile and Train the Model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10, validation_data=test_generator)
The model is trained for 10 epochs and validated on the test dataset.

5. Evaluate the Model

loss, acc = model.evaluate(test_generator)
Displays final test accuracy.

6. Upload and Predict a New Image
Upload any cat or dog image from your local system and the model will predict its class.

uploaded = files.upload()
🧠 Prediction Output
If prediction < 0.5 → Cat

If prediction ≥ 0.5 → Dog

📦 Requirements
Install the following Python packages (if not using Google Colab):

pip install tensorflow numpy pillow
📌 Notes
This model is designed for binary classification (cats vs dogs).

Ensure images are clearly labeled and well-distributed for best results.

You can increase the number of epochs or add augmentation to improve accuracy.

✅ Example Output
Extracted folders: ['train', 'test']
Found 2000 images belonging to 2 classes.
Found 1000 images belonging to 2 classes.
Epoch 1/10...
Test Accuracy: 68.40%
Prediction: Dog 🐶
