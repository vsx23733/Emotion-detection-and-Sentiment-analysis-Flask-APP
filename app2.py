from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
from werkzeug.utils import secure_filename
import cv2
from keras.models import load_model
import numpy as np
import os
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import librosa
import tensorflow as tf
import time
import mediapipe as mp

app2 = Flask(__name__)
app2.config['UPLOAD_FOLDER'] = 'uploads'
app2.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'txt', 'mp3', 'avi'}

# Make sure the upload folder exists
os.makedirs(app2.config['UPLOAD_FOLDER'], exist_ok=True)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# Load pre-trained emotion detection model (expects input shape: (None, 64, 64, 1))
model = load_model(r'emotion_detection_model.hdf5', compile=False)

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# OpenCV video capture for realtime webcam (0 = default camera)
camera = cv2.VideoCapture(0)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app2.config['ALLOWED_EXTENSIONS']


def preprocess_image_array(gray_frame):
    """
    Takes a grayscale frame and returns a tensor of shape (1, 64, 64, 1)
    ready for the model.
    """
    # Resize to 64x64 because the model expects (64, 64, 1)
    image = cv2.resize(gray_frame, (64, 64))

    # Convert to float and normalize
    image = image.astype("float32") / 255.0

    # Add channel dimension -> (64, 64, 1)
    image = np.expand_dims(image, axis=-1)

    # Add batch dimension -> (1, 64, 64, 1)
    image = np.expand_dims(image, axis=0)

    return image


def detect_emotion(image_path):
    # For file uploads (existing behavior)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess_image_array(image)

    predictions = model.predict(image)
    emotion_index = np.argmax(predictions)
    emotion = emotion_labels[emotion_index]

    return emotion


def detect_emotion_from_frame(frame):
    """
    For realtime webcam: takes a BGR frame, returns predicted emotion string.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = preprocess_image_array(gray)

    predictions = model.predict(image)
    emotion_index = np.argmax(predictions)
    emotion = emotion_labels[emotion_index]

    return emotion


def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    sentiment_label = 'Positive' if sentiment_scores['compound'] >= 0 else 'Negative'
    return sentiment_label


def gen_frames():
    """
    Generator that reads frames from the webcam, runs emotion detection,
    draws the label, and yields JPEG bytes for streaming.
    """
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Get emotion prediction for this frame
        emotion = detect_emotion_from_frame(frame)

        # Draw the emotion label on the frame
        cv2.putText(
            frame,
            f'Emotion: {emotion}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # MJPEG streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app2.route('/', methods=['GET', 'POST'])
def index():
    # Original page with upload form etc. (unchanged behavior)
    if request.method == 'POST':
        operation = request.form.get('operation')

        if operation == 'emotion':
            # Check if file was uploaded
            if 'file' not in request.files:
                return render_template('index.html', error='No file uploaded.')

            file = request.files['file']

            # Check if file has a valid extension
            if file.filename == '' or not allowed_file(file.filename):
                return render_template('index.html', error='Invalid file selected.')

            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app2.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Perform emotion detection on the uploaded image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                emotions = detect_emotion(file_path)

                # Render the result template with the emotion detected
                return render_template('result.html', filename=filename, emotion=emotions)

        elif operation == 'sentiment':
            # Check if file was uploaded
            if 'file' not in request.files:
                return render_template('index.html', error='No file uploaded.')

            file = request.files['file']

            # Check if file has a valid extension
            if file.filename == '' or not allowed_file(file.filename):
                return render_template('index.html', error='Invalid file selected.')

            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app2.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if filename.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()

                sentiment = analyze_sentiment(text)

                # Render the result template with the sentiment detected
                return render_template('result.html', filename=filename, emotion=None, sentiment=sentiment)

    return render_template('index.html')


@app2.route('/realtime')
def realtime():
    """
    Page that displays the realtime webcam stream.
    In your template, you'll point an <img> to /video_feed.
    """
    return render_template('realtime.html')


@app2.route('/video_feed')
def video_feed():
    """
    Video streaming route. Put this as the src of an <img> tag.
    """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        app2.run(host='0.0.0.0', port=5000)
    finally:
        # Release the camera when the server stops
        camera.release()
        cv2.destroyAllWindows()
