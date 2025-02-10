import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
import keras
import pickle
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array
# from keras.applications.mobilenet_v2 import preprocess_input
import time

# Load pre-trained model
# model = tf.keras.models.load_model('my_model.keras')

#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tf.keras.models import Sequential
# from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
# from keras.utils import np_utils
from sklearn.model_selection import train_test_split
#from keras.layers import Input
data = pd.read_csv('fer2013.csv')
print(data.head())

pixels = data['pixels'].tolist()
images = np.array([np.fromstring(pixel, dtype=int, sep=' ') for pixel in pixels])
images = images.reshape(images.shape[0], 48, 48, 1)
images = images.astype('float32') / 255.0

# Extract labels
emotions = pd.get_dummies(data['emotion']).values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, emotions, test_size=0.2, random_state=42)
model =  tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# history = model.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test))

#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Early stopping to stop training when validation loss stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Save the best model during training
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Reduce learning rate when a metric has stopped improving
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.001)

# Fit the model with the callbacks
history = model.fit(
    X_train, y_train,
    batch_size=128,  # Increased batch size
    epochs=2,       # Reduced number of epochs
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)
# Save the trained model
# model.save('my_emotion_model.keras')
# model.save('my_model.keras')
# import pickle

# Save the trained model as a .pkl file
with open('emotion_model.pkl', 'wb') as file:
    pickle.dump(model, file)


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

predictions = model.predict(X_test)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i].reshape(48, 48), cmap=plt.cm.gray)
    plt.xlabel(f"Actual: {np.argmax(y_test[i])}\nPredicted: {np.argmax(predictions[i])}")
plt.show()

# model = tf.keras.models.load_model('my_model.keras')

# Emotions labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Load the image
image_path = 'image.jpg'
frame = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Load your face detection model (assuming face_cascade is already initialized)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Iterate over detected faces
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

# Display the resulting image
cv2.imshow('Image', frame)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()










