from flask import Flask, render_template, request, redirect, url_for, send_from_directory
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

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# Load pre-trained emotion detection model
model = load_model(r'C:\Users\axelo\anaconda3\Lib\site-packages\keras\models\emotion_detection_model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app2.config['ALLOWED_EXTENSIONS']

def detect_emotion(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # Predict emotions
    predictions = model.predict(image)
    emotion_index = np.argmax(predictions)
    emotion = emotion_labels[emotion_index]

    return emotion

def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    sentiment_label = 'Positive' if sentiment_scores['compound'] >= 0 else 'Negative'
    return sentiment_label


@app2.route('/', methods=['GET', 'POST'])
def index():
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
            if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
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

            if filename.endswith('.txt'):
                with open(file_path, 'r') as f:
                    text = f.read()

                sentiment = analyze_sentiment(text)

                # Render the result template with the sentiment detected
                return render_template('result.html', filename=filename, emotion=None, sentiment=sentiment)


    return render_template('index.html')

if __name__ == '__main__':
    app2.run(host='0.0.0.0', port=5000)
