import cv2
import tensorflow as tf
import numpy as np

# Load pre-trained model (e.g., a MobileNetV2 or an action recognition model)
model = tf.keras.models.load_model('your_action_recognition_model.h5')

# Load class labels (action categories)
class_labels = ['Action1', 'Action2', 'Action3', 'Action4']

# Open video file or webcam
cap = cv2.VideoCapture(0)  # 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame (resize, normalize, etc.)
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to model's input shape
    input_frame = np.expand_dims(resized_frame, axis=0) / 255.0  # Normalize the frame

    # Perform action prediction
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions, axis=1)
    action = class_labels[predicted_class[0]]

    # Display the action on the frame
    cv2.putText(frame, action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with predicted action
    cv2.imshow('Action Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
