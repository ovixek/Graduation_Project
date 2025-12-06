
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import streamlit as st
from PIL import Image
# Initialize pygame mixer and load the alarm sound
mixer.init()
try:
    sound = mixer.Sound('alarm.wav')
except Exception as e:
    st.error("🚨 Alarm sound file 'alarm.wav' not found. Please ensure the file exists in the working directory.")
    st.stop()

# Load pre-trained Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the trained model for eye state detection
try:
    model = load_model('final_model.h5')
except Exception as e:
    st.error("❌ Model file 'final_model.h5' not found. Please ensure the model exists in the 'models' directory.")
    st.stop()

# Streamlit interface setup
st.set_page_config(page_title="Drowsiness Detection App", page_icon="🚗", layout="centered")
st.title("🚗 Real-Time Drowsiness Detection App")
st.subheader("Stay Alert, Stay Safe!")
st.write("This application uses AI-powered eye and face detection to monitor drowsiness in real time.")

# Initialize session states
if "run_model" not in st.session_state:
    st.session_state.run_model = False  # Control when detection runs
    st.session_state.Score = 0          # Track the drowsiness score

# Utility: Reset the score and stop the alarm
def reset_score_and_stop_alarm():
    st.session_state.Score = 0
    try:
        sound.stop()
    except:
        pass

# Layout: Buttons for starting and stopping the detection
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Detection"):
        st.session_state.run_model = True
        st.success("Detection started! Look into the camera.")

with col2:
    if st.button("⏹ Stop Detection"):
        st.session_state.run_model = False
        reset_score_and_stop_alarm()
        st.info("Detection stopped.")

# Persistent score display
score_placeholder = st.empty()

# Webcam logic for live detection
if st.session_state.run_model:
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Unable to access the webcam. Please check your device settings.")
        st.session_state.run_model = False
        st.stop()

    frame_placeholder = st.empty()  # Placeholder for the live video feed

    try:
        while st.session_state.run_model and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture webcam frame.")
                break

            # Convert to grayscale for Haar cascades
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw face rectangle

            # Detect eyes within the face
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
            for (ex, ey, ew, eh) in eyes:
                # Extract eye region
                eye = frame[ey:ey + eh, ex:ex + ew]

                # Preprocess the eye region for the model
                eye = cv2.resize(eye, (80, 80))
                eye = eye / 255.0  # Normalize pixel values
                eye = eye.reshape(80, 80, 3)
                eye = np.expand_dims(eye, axis=0)

                # Predict eye state using the model
                prediction = model.predict(eye)

                # Update score based on prediction
                if prediction[0][0] > 0.30:  # Eye is closed
                    st.session_state.Score += 1
                    if st.session_state.Score > 15:  # Trigger alarm if score exceeds threshold
                        try:
                            sound.play()
                        except:
                            st.warning("🚨 Alarm sound could not be played.")
                elif prediction[0][1] > 0.90:  # Eye is open
                    st.session_state.Score -= 1
                    st.session_state.Score = max(st.session_state.Score, 0)

            # Update the score display
            score_placeholder.markdown(f"### Drowsiness Score: **{st.session_state.Score}**")

            # Convert BGR (OpenCV) to RGB for Streamlit display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(rgb_frame)
            frame_placeholder.image(frame_image, use_container_width=True)

            # Stop if detection is turned off
            if not st.session_state.run_model:
                break
    finally:
        # Ensure resources are released even if an error occurs
        cap.release()
        reset_score_and_stop_alarm()
else:
    st.info("Press the ▶️ Start Detection button to begin.")

# Footer
st.markdown("""
---
### 💡 Tips for Best Results
1. Ensure proper lighting in the environment.
2. Position your face clearly in front of the webcam.
3. Avoid obstructions like glasses or hair covering the eyes.

##### Be Aware, Be Safe!
""")