# AI-Generated vs Real Image Detector

## Project Overview

This project aims to develop a deep learning model capable of distinguishing between AI-generated images and real photographs. The model uses Error Level Analysis (ELA) as a preprocessing step to highlight potential artifacts in images that may indicate artificial generation.

## Table of Contents

1. [Dataset](#dataset)
2. [Preprocessing](#preprocessing)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Usage](#usage)
7. [Dependencies](#dependencies)

## Dataset

The project uses two datasets:

1. DALL-E Recognition Dataset
   - Real images: `/kaggle/input/dalle-recognition-dataset/real`
   - Fake images: `/kaggle/input/dalle-recognition-dataset/fakeV2/fake-v2`

2. AI-Generated Images vs Real Images Dataset
   - Real images: `/kaggle/input/ai-generated-images-vs-real-images/RealArt/RealArt`
   - AI-generated images: `/kaggle/input/ai-generated-images-vs-real-images/AiArtData/AiArtData`

## Preprocessing

1. **Image Resizing**: All images are resized to 224x224 pixels.
2. **Error Level Analysis (ELA)**: This technique is applied to highlight potential artifacts in images.
3. **Data Augmentation**: The `ImageDataGenerator` is used to apply real-time data augmentation during training.

## Model Architecture

The model uses a Convolutional Neural Network (CNN) architecture:

- Input shape: (224, 224, 3)
- 3 Convolutional layers with ReLU activation and MaxPooling
- Flatten layer
- Dense layer with 512 units and ReLU activation
- Dropout layer (0.5)
- Output Dense layer with sigmoid activation for binary classification

## Training

- The dataset is split into training and validation sets (80% / 20%).
- The model is trained for 2 epochs using the Adam optimizer.
- Binary cross-entropy is used as the loss function.
- The trained model is saved as 'idk.h5'.

## Evaluation

The model's performance can be evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

(Note: The provided code doesn't explicitly calculate these metrics, but they are mentioned in the imports.)

## Usage

To use the trained model for prediction:

1. Load the model:
   ```python
   model = tf.keras.models.load_model('idk.h5')
   ```

2. Preprocess the image:
   ```python
   def preprocess_image(image):
       resized_image = cv2.resize(image, (224, 224))
       resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
       ela_image = error_level_analysis(resized_image_rgb)
       normalized_image = ela_image / 255.0
       return normalized_image
   ```

3. Make a prediction:
   ```python
   def predict_image(image_path):
       image = cv2.imread(image_path)
       processed_image = preprocess_image(image)
       input_image = np.expand_dims(processed_image, axis=0)
       prediction = model.predict(input_image)[0][0]
       class_name = "AI-generated" if prediction > 0.5 else "Real"
       confidence = prediction if prediction > 0.5 else 1 - prediction
       return f"{class_name} (Confidence: {confidence:.2f})"
   ```

## Dependencies

- NumPy
- TensorFlow
- Keras
- OpenCV (cv2)
- scikit-image
- scikit-learn
- Matplotlib
- Pandas

To install the required packages, run:
```bash
!pip install numpy tensorflow opencv-python scikit-image scikit-learn matplotlib pandas
```

## Conclusion

This project demonstrates the application of deep learning techniques to the challenging task of distinguishing between AI-generated and real images, leveraging Error Level Analysis as a preprocessing step to enhance the model's ability to detect potential artifacts in artificially generated images.
