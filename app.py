import streamlit 
import pickle
import cv2
import numpy as np
import tensorflow as tf

# Load the model from the .pkl file
with open('emotion_model1.pkl', 'rb') as file:
    model = pickle.load(file)

# Emotion labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

streamlit.title('Facial Emotion Recognition')

# Upload an image
uploaded_file = streamlit.file_uploader("Choose an image...", type=["jpeg","jpg","jfif"])

if uploaded_file is not None:
    # Convert the uploaded file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each face found
    for (x, y, w, h) in faces:
        # Extract the ROI (Region of Interest) containing the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        # Preprocess the image for prediction
        roi = roi_gray.astype('float') / 255.0
        roi = tf.keras.preprocessing.image.img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Make predictions
        preds = model.predict(roi)[0]
        label = emotion_labels[preds.argmax()]
        
        # Draw bounding box and label on the image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Convert the image to RGB before displaying
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the image with detected faces and emotions
    streamlit.image(frame, caption='Uploaded Image', use_column_width=True)
