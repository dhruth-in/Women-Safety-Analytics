import cv2
import numpy as np
from collections import OrderedDict
from deepface import DeepFace

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

# Load MobileNet-SSD
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture('video1.mp4')
ct = CentroidTracker()

id_to_gender = {}  # Maps object_id to gender
id_to_emotion = {}  # Maps object_id to emotion

# Resize the window
cv2.namedWindow("People Detection, Tracking & Emotion and Gender Counting", cv2.WINDOW_NORMAL)
cv2.resizeWindow("People Detection, Tracking & Emotion and Gender Counting", 640, 480)  # Set the window size to 640x480

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    centroids = []
    bounding_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            centroids.append(centroid)
            bounding_boxes.append((x1, y1, x2, y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    objects = ct.update(np.array(centroids))

    for ((object_id, centroid), (x1, y1, x2, y2)) in zip(objects.items(), bounding_boxes):
        if object_id not in id_to_gender:
            face_img = frame[y1:y2, x1:x2]
            try:
                result = DeepFace.analyze(face_img, actions=['gender', 'emotion'], enforce_detection=False)
                
                # Check if result is a list and has data
                if isinstance(result, list) and len(result) > 0:
                    dominant_gender = result[0].get('dominant_gender', 'Unknown')
                    dominant_emotion = result[0].get('dominant_emotion', 'Unknown')
                elif isinstance(result, dict):
                    dominant_gender = result.get('dominant_gender', 'Unknown')
                    dominant_emotion = result.get('dominant_emotion', 'Unknown')
                else:
                    dominant_gender = 'Unknown'
                    dominant_emotion = 'Unknown'

                id_to_gender[object_id] = dominant_gender
                id_to_emotion[object_id] = dominant_emotion
            except Exception as e:
                print(f"Error analyzing gender and emotion: {e}")
                continue

        gender_label = id_to_gender.get(object_id, 'Unknown')
        emotion_label = id_to_emotion.get(object_id, 'Unknown')
        text = f"ID {object_id} ({gender_label}, {emotion_label})"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.circle(frame, centroid, 4, (255, 255, 255), -1)

    # Display the number of people detected in the frame
    people_count = len(objects)  # Count the number of people (objects) in the frame
    cv2.putText(frame, f"People: {people_count}", (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3) 

    # Resize the frame to smaller size before displaying
    small_frame = cv2.resize(frame, (7200, 620))  # Resize to 640x480

    cv2.imshow("People Detection, Tracking & Emotion and Gender Counting", small_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
