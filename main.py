import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam

# Initialize Flask
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define available actions
actions = [
    'apple', 'basketball', 'cabbage', 'cherry', 'football', 'hello', 'hockey',
    'no', 'potato', 'see_you_later', 'sorry', 'take_care', 'tennis', 'thank_you',
    "what's_up", 'yes'
]

# Load Mediapipe Holistic model once
mp_holistic = mp.solutions.holistic

# Load the trained LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='tanh', input_shape=(60, 7806)),
    LayerNormalization(),
    Dropout(0.3),
    LSTM(128, return_sequences=True, activation='tanh'),
    LayerNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=False, activation='tanh'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.load_weights('first.keras')

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def extract_hand_crop(frame, hand_landmarks):
    if hand_landmarks:
        h, w, _ = frame.shape
        x_min, y_min, x_max, y_max = w, h, 0, 0
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)
        padding = 50
        x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
        x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)
        hand_crop = frame[y_min:y_max, x_min:x_max]
        return cv2.resize(hand_crop, (32, 32)) if hand_crop.size > 0 else np.zeros((32, 32, 3))
    return np.zeros((32, 32, 3))
def predict(video_path, target_label, model, duration=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = duration * fps
    predictions, sequence = [], []
    threshold_count = 0  # To count consecutive frames above 50%

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        frame_count = 0
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            image, holistic_results = mediapipe_detection(frame, holistic)
            landmarks = extract_keypoints(holistic_results).flatten()
            left_hand_crop = extract_hand_crop(frame, holistic_results.left_hand_landmarks)
            right_hand_crop = extract_hand_crop(frame, holistic_results.right_hand_landmarks)
            sequence.append(np.concatenate([landmarks, left_hand_crop.flatten(), right_hand_crop.flatten()]))
            sequence = sequence[-60:]

            if len(sequence) == 60:
                out = model.predict(np.array([sequence]))[0]  # Get probabilities
                predicted_label = np.argmax(out)
                confidence = out[target_label]  # Get confidence of target label

                print(f"Frame {frame_count}: Predicted={predicted_label}, Confidence={confidence:.2f}")

                # if predicted_label == target_label and confidence >=  0.01:
                if confidence >=  0.01:
                    threshold_count += 1
                else:
                    threshold_count = 0  # Reset if confidence drops

                if threshold_count >= 5:  # If label has been detected 5 times above 50% confidence
                    cap.release()
                    return True

            frame_count += 1
        cap.release()

    return False


@app.route("/predict", methods=["POST"])
def predict_label():
    if "video" not in request.files or "label" not in request.form:
        return jsonify({"error": "Missing video file or label"}), 400
    video = request.files["video"]
    try:
        target_label = int(request.form["label"])
    except ValueError:
        return jsonify({"error": "Label must be an integer."}), 400
    if not (0 <= target_label < len(actions)):
        return jsonify({"error": f"Invalid label. Must be in range 0-{len(actions)-1}"}), 400
    filename = secure_filename(video.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    video.save(filepath)
    result = predict(filepath, target_label, model)
    os.remove(filepath)  # Delete video after processing
    return jsonify({"label_found": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
