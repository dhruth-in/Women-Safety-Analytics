from builtins import Exception, exit, isinstance, list, ord, print
import cv2
from deepface import DeepFace
from gender_counter import GenderCounter

# Use webcam (0) or video file
cap = cv2.VideoCapture('video1.mp4')  # Use "video1.mp4" instead of 0 to use a video file

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    try:
        results = DeepFace.analyze(frame, actions=['gender', 'emotion'], enforce_detection=False)

        # Ensure results is a list
        if not isinstance(results, list):
            results = [results]

        for res in results:
            region = res['region']
            gender = res['gender']
            emotion = res['dominant_emotion']

            # Draw rectangle around face
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Label with gender and emotion
            label = f"{gender}, {emotion}"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    except Exception as e:
        print(f"Error: {e}")

    cv2.imshow("Gender and Emotion Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
