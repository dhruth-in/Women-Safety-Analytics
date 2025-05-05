# Women Safety Analysis – Real-Time Scene Understanding

## 📌 Overview

**Women Safety Analysis** is a real-time computer vision system designed to enhance situational awareness and contribute to women’s safety. This project leverages **hand gesture recognition**, **gender and emotion detection**, and **people counting** using live camera input. It intelligently interprets the scene and offers critical insights to help identify potential threats or emergency gestures.

---

## 🎯 Features

* ✋ **Hand Gesture Detection** – Identifies predefined hand gestures using MediaPipe and custom classifiers (e.g., Help, Stop, Peace).
* 😊 **Emotion Recognition** – Detects emotions like Happy, Sad, Angry, Neutral using facial expressions.
* 🚻 **Gender Detection** – Classifies detected faces as male or female.
* 🧍‍♀️🧍 **People Counting** – Detects and displays the number of people present in the frame.
* ⚡ **Real-Time Processing** – Efficient performance with live webcam feed using OpenCV.

---

## 🧠 Technologies Used

* **Python**
* **OpenCV**
* **MediaPipe**
* **TensorFlow/Keras** (for emotion and gender detection models)
* **NumPy**
* **Custom Machine Learning Models** (gesture and point history classification)

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/women-safety-analysis.git
cd women-safety-analysis
```

### 2. Install Dependencies

We recommend using a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Download Models

Ensure the following models are placed in the correct directories:

* `gesture_classifier.model`
* `emotion_model.h5`
* `gender_model.h5`
* MediaPipe uses pre-built models (auto-downloaded)

---

## 🚀 How to Run

```bash
python main.py
```

This will activate your webcam and start real-time analysis.

---

## 🎯 How It Works

1. **Capture Input** – Webcam feed is processed frame-by-frame.
2. **Face Detection** – Faces are detected using OpenCV or MediaPipe.
3. **Emotion & Gender** – Each face is passed through emotion and gender classifiers.
4. **Hand Landmark Detection** – Hands are detected, keypoints are classified into predefined gestures.
5. **Scene Output** – Overlay of:

   * Detected gender and emotion
   * Gesture (if any)
   * Number of people in frame

---

## 🧪 Use Cases

* Smart surveillance in public places
* Emergency signal detection in isolated areas
* Real-time awareness apps for safety and monitoring

---

## 🙋‍♀️ Contributors

* **Your Name** – DHRUTHI N (https://github.com/dhruth-in)
