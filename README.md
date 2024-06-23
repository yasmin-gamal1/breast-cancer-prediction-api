# Breast Cancer Prediction API

## Overview

This API provides functionality to predict breast cancer based on uploaded images. The model used is a pre-trained neural network that can classify whether the image indicates the presence of breast cancer.

## Requirements

- Python 3.6+
- Flask
- Flask-CORS
- TensorFlow
- Pillow
- NumPy
- Matplotlib
- Plotly
- Seaborn
- OpenCV

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/breast-cancer-prediction-api.git
   cd breast-cancer-prediction-api
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place the model file `my_model.h5` in the project directory**

4. **Run the API**
   ```bash
   python app.py
   ```

## Endpoints

### Home

**URL:** `/`  
**Method:** `GET`  
**Description:** Returns a welcome message.

**Response:**
```json
"Welcome to the Breast Cancer Prediction API!"
```

### Predict

**URL:** `/predict`  
**Method:** `POST`  
**Description:** Accepts an image file and returns a prediction.

**Request:**

- **Headers:** `Content-Type: multipart/form-data`
- **Body:** A form-data with a key `file` containing the image file.

**Response:**

- **Success (200):**
  ```json
  {
    "prediction": 1
  }
  ```
  where `prediction` is the class label predicted by the model.

- **Error (400):**
  ```json
  {
    "error": "Error message"
  }
  ```

## Usage

### Using Postman

1. **Open Postman**.

2. **Create a new POST request**.

3. **Set the URL** to `http://localhost:5000/predict`.

4. **Go to the `Body` tab**, select `form-data`, and add a key `file`.

5. **Select the type as `File`** and upload an image file.

6. **Send the request** and view the prediction in the response.

### Sample Code

Here's how you can use the API programmatically with Python's `requests` library:

```python
import requests

url = 'http://localhost:5000/predict'
file_path = 'path_to_your_image.jpg'

with open(file_path, 'rb') as f:
    response = requests.post(url, files={'file': f})

print(response.json())
```

## Model Information

The model used in this API is a Convolutional Neural Network (CNN) trained to classify breast cancer images. The model architecture is as follows:

- Input layer with a shape of (50, 50, 3)
- 4 Convolutional layers with ReLU activation and max pooling
- Flatten layer
- Dense layer with ReLU activation
- Output layer with softmax activation

The model was trained using the following steps:

1. **Data Loading and Preprocessing:**
    ```python
    import pandas as pd
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import plotly.express as px
    import seaborn as sns
    import glob
    import random
    import os
    from keras.preprocessing import image
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    ```

2. **Data Visualization:**
    ```python
    bar = px.bar(data_frame=data_insight_1, x='state of cancer', y='Numbers of Patients', color='state of cancer')
    bar.update_layout(title_text='Number of Patients with cancer (1) and patients with no cancer (0)', title_x=0.5)
    bar.show()
    ```

3. **Data Splitting:**
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)
    ```

4. **Model Building:**
    ```python
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(50, 50, 3)),
        tf.keras.layers.MaxPooling2D(strides=2),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=41, batch_size=75)
    ```

5. **Model Evaluation:**
    ```python
    model.evaluate(X_test, y_test)
    ```

6. **Confusion Matrix:**
    ```python
    from sklearn.metrics import confusion_matrix
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(y_test, axis=1)

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    f, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="BuPu", linecolor="gray", fmt='.1f', ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    ```

7. **Save the Model:**
    ```python
    model.save_weights('./checkpoints/my_checkpoint')
    ```

This model has been trained and tested to ensure high accuracy in predicting breast cancer from medical images.
