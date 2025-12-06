## Augmented Road Safety: Real-Time Drowsiness Detection and Automated Alert System

This repository contains the implementation of a real-time driver drowsiness detection system developed using deep learning and computer vision techniques. The system identifies early signs of fatigue by monitoring eye states through a webcam and triggers an alert when drowsiness is detected.

# 1. Introduction

Driver fatigue is a leading cause of road accidents. This project proposes a non-invasive, software-based solution capable of detecting drowsiness in real time using a standard webcam, deep learning models, and computer vision techniques.

The system is designed to:

Continuously monitor the driver’s eyes

Classify eye states as open or closed

Calculate a drowsiness score

Trigger an alarm when the score exceeds a threshold

# 2. Features

Non-intrusive, webcam-based detection

Real-time eye and face tracking using Haar cascades

Deep learning classification using InceptionV3

Automated alarm triggering when drowsiness is detected

Streamlit-based graphical user interface

Robust performance under varying lighting and head orientations

# 3. System Architecture
3.1 Block Diagram

Pipeline includes:

Data Collection

Preprocessing

Model Training

Real-Time Face and Eye Detection

Eye State Prediction

Drowsiness Scoring

Alarm Triggering

# 4. Dataset

The dataset used contains:

84,898 eye images

37 subjects (33 male, 4 female)

Annotations include:

Subject ID

Eye state (open/closed)

Presence of glasses

Gender

Reflection level

Lighting condition

# 5. Methodology

5.1 Data Preprocessing

Normalization

Image augmentation (rotation, shifting, zooming, shear)

Resizing images to 80x80

Splitting into training, validation, and test sets

5.2 Model Architecture

Pretrained InceptionV3 as feature extractor

Custom dense layers added for binary classification

Softmax activation for open/closed eye prediction

Transfer learning used by freezing base model layers


5.3 Model Training

Batch size: 8

Epochs: 10

Optimizer: Adam

Loss function: Categorical Crossentropy

EarlyStopping and ModelCheckpoint used

# 6. Real-Time Detection System

6.1 Detection Pipeline

Capture frame via webcam

Convert to grayscale

Detect face using Haar cascades

Detect eyes inside face region

Preprocess eye ROI

Predict eye state using trained model

Update drowsiness score

Trigger alarm if score exceeds threshold (15)

6.2 Libraries Used

OpenCV

TensorFlow / Keras

NumPy

Pygame (for alarm sound)

Streamlit (for UI)

6.3 Streamlit UI

Start/Stop detection buttons

Live frame display

Drowsiness score indicator

Alerts shown on screen

# 7. Performance Evaluation
7.1 Evaluation Metrics
Metric	            Value
Test Accuracy	     93.98%
Test Loss	         18.89%
Precision	         0.9295
Recall	           0.9500
F1-Score	         0.9390
Specificity	       0.9280
ROC AUC	           0.9440

7.2 Confusion Matrix

True Positives:     475
True Negatives:     464
False Positives:     36
False Negatives:     25

# 8. Results

The system successfully:

Detects face and eye regions in real time

Computes drowsiness score dynamically

Triggers an alarm when the score crosses the threshold

Displays real-time visual feedback to the user

Observation images are shown in the report.

# 9. Installation and Usage

9.1 Requirements Installation
pip install -r requirements.txt

9.2 Run the Streamlit Application
streamlit run app.py

# Contributors

Abhishek Pujari
Angsuman Bhuyan
Debanga Baruah
Under the guidance of Dr. Nandita Deka
