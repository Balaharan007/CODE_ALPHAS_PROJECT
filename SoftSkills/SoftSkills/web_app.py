from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import json
import os
import base64
import threading
import time
import random
from collections import Counter
from io import BytesIO
import tempfile
import librosa
import sounddevice as sd
import scipy.io.wavfile as wav
import speech_recognition as sr
from textblob import TextBlob
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

# Retrieve values from .env
FER_MODEL_PATH = os.getenv("FER_MODEL_PATH")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", 22050))
AUDIO_DURATION = int(os.getenv("AUDIO_DURATION", 5))
HAAR_CASCADE_PATH = os.getenv("HAAR_CASCADE_PATH")

# Load models
face_emotion_model = load_model(FER_MODEL_PATH)
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

app = Flask(__name__)

# Global variables
is_recording = False
current_frame = None
recording_data = {
    "frames": [],
    "detected_emotions": [],
    "emotion_confidence_scores": [],
    "emotion_timestamps": []
}

def analyze_voice_tone(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path)
    
    # Extract pitch using YIN algorithm
    pitch = librosa.yin(y, fmin=50, fmax=300)
    
    # Calculate intensity (volume)
    intensity = np.abs(y).mean()
    
    # Get average pitch, excluding NaN values
    avg_pitch = np.nanmean(pitch)
    
    # Extract additional features for a more comprehensive analysis
    # Spectral centroid - brightness of sound
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    
    # Spectral bandwidth - width of the spectral band
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    
    # Root Mean Square Energy - volume over time
    rms = librosa.feature.rms(y=y)[0].mean()
    
    # Zero Crossing Rate - how often signal changes sign
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0].mean()
    
    # Tempo estimation
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    
    # Return a more detailed voice analysis
    return {
        "pitch": avg_pitch,
        "intensity": intensity,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "rms": rms,
        "zero_crossing_rate": zero_crossing_rate,
        "tempo": tempo
    }

def analyze_speech_sentiment(audio_path):
    recognizer = sr.Recognizer()
    audio_text = ""

    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            try:
                audio_text = recognizer.recognize_google(audio)
                print(f"Transcribed Text: {audio_text}")
            except sr.UnknownValueError:
                print("Could not understand audio - no speech detected")
                return "Neutral", "", 0.0, 0.5  # Return default values for sentiment, text, score, subjectivity
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                return "Neutral", "", 0.0, 0.5
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return "Neutral", "", 0.0, 0.5

    # Analyze sentiment with TextBlob
    try:
        blob = TextBlob(audio_text)
        sentiment_score = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment category
        if sentiment_score > 0.1:
            sentiment = "Positive"
        elif sentiment_score < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        print(f"Speech sentiment: {sentiment} (score: {sentiment_score:.2f}, subjectivity: {subjectivity:.2f})")
        
        return sentiment, audio_text, sentiment_score, subjectivity
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return "Neutral", audio_text, 0.0, 0.5

def calculate_soft_skill_score(emotion_data, voice_data, sentiment_data):
    # Extract emotion info
    dominant_emotion = emotion_data["dominant_emotion"]
    emotion_distribution = emotion_data["distribution"]
    emotion_stability = emotion_data["stability"]
    
    # Base emotion scores
    emotion_scores = {
        "Happy": 9, 
        "Neutral": 7, 
        "Surprise": 6, 
        "Sad": 4, 
        "Angry": 3, 
        "Fear": 2, 
        "Disgust": 1
    }
    
    # Extract voice metrics
    pitch = voice_data["pitch"]
    intensity = voice_data["intensity"]
    spectral_centroid = voice_data["spectral_centroid"]
    tempo = voice_data["tempo"]
    
    # Extract sentiment data
    sentiment = sentiment_data["sentiment"]
    sentiment_score = sentiment_data["score"]
    subjectivity = sentiment_data["subjectivity"]
    
    # Calculate base scores
    emotion_score = emotion_scores.get(dominant_emotion, 5)
    
    # Adjust emotion score based on distribution and stability
    emotion_variety_bonus = min(3, len([e for e, v in emotion_distribution.items() if v > 0.1]))
    emotion_score = min(10, emotion_score + (emotion_variety_bonus * 0.3))
    
    # Stability can either be good or bad depending on context
    # Too stable might mean monotonous, too unstable might mean erratic
    stability_adjustment = 0
    if emotion_stability < 0.3:  # Very unstable
        stability_adjustment = -1
    elif emotion_stability > 0.8:  # Very stable
        if dominant_emotion in ["Happy", "Neutral"]:
            stability_adjustment = 1
        else:
            stability_adjustment = -0.5
            
    emotion_score = max(1, min(10, emotion_score + stability_adjustment))
    
    # Calculate sentiment score
    sentiment_base_scores = {"Positive": 9, "Neutral": 6, "Negative": 3}
    sentiment_score_value = sentiment_base_scores.get(sentiment, 5)
    
    # Adjust sentiment score based on subjectivity (more subjective can be more engaging)
    sentiment_score_value = min(10, sentiment_score_value + (subjectivity * 2))
    
    # Calculate pitch score - preferred range is gender and individual dependent
    # Here we use a simplified approach
    pitch_score = 10 if 100 <= pitch <= 250 else 5
    
    # Intensity score - moderate intensity is usually better
    intensity_score = 10 if 0.02 <= intensity <= 0.1 else 5
    
    # Voice variability score based on spectral features and tempo
    voice_variability = (spectral_centroid / 5000) * 10  # Normalize to 0-10 scale
    tempo_score = min(10, (tempo / 180) * 10)  # Normalize to 0-10 scale
    
    # Calculate final score with weighted components
    final_score = (
        (emotion_score * 0.25) +         # Facial emotion weight
        (sentiment_score_value * 0.25) +  # Speech sentiment weight
        (pitch_score * 0.15) +            # Voice pitch weight
        (intensity_score * 0.15) +        # Voice intensity weight
        (voice_variability * 0.1) +       # Voice variability weight
        (tempo_score * 0.1)               # Speech tempo weight
    )
    
    # Ensure final score is in range 1-10
    final_score = max(1, min(10, final_score))
    
    return {
        "emotion_score": round(emotion_score, 1),
        "sentiment_score": round(sentiment_score_value, 1),
        "pitch_score": round(pitch_score, 1),
        "intensity_score": round(intensity_score, 1),
        "voice_variability_score": round(voice_variability, 1),
        "tempo_score": round(tempo_score, 1),
        "final_score": round(final_score, 1)
    }

def get_improvement_suggestions(score, emotion_data, voice_data, sentiment_data):
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    # Create a more specific prompt based on the detailed analysis
    dominant_emotion = emotion_data["dominant_emotion"]
    sentiment = sentiment_data["sentiment"]
    pitch = voice_data["pitch"]
    intensity = voice_data["intensity"]
    
    prompt_text = f"""
    A user has a soft skills assessment with the following results:
    - Overall Score: {score}/10
    - Dominant Facial Expression: {dominant_emotion}
    - Speech Sentiment: {sentiment}
    - Voice Pitch: {pitch:.2f} Hz
    - Voice Intensity: {intensity:.4f}
    
    Based on these specific metrics, provide tailored suggestions to improve their:
    1. Communication effectiveness
    2. Emotional expression
    3. Vocal delivery
    4. Overall confidence and engagement
    
    Focus on practical, actionable advice that addresses their specific results.
    """

    prompt = {
        "contents": [{
            "parts": [{
                "text": prompt_text
            }]
        }]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{endpoint}?key={GEMINI_API_KEY}", json=prompt, headers=headers)

    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Unable to fetch suggestions. Error: {response.text}"

def process_frame(frame):
    global current_frame, recording_data
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    detected_emotion = "Neutral"
    emotion_scores = np.zeros(len(emotion_labels))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        roi_gray = np.expand_dims(roi_gray, axis=-1)
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.repeat(roi_gray, 3, axis=-1)
        roi_gray = roi_gray / 255.0

        predictions = face_emotion_model.predict(roi_gray)
        emotion_scores = predictions[0]
        max_index = np.argmax(emotion_scores)
        detected_emotion = emotion_labels[max_index]

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # If recording, save detected emotion and confidence scores
    if is_recording and len(faces) > 0:
        recording_data["detected_emotions"].append(detected_emotion)
        recording_data["emotion_confidence_scores"].append(emotion_scores)
        recording_data["emotion_timestamps"].append(time.time())
    
    # Update the current frame
    current_frame = frame
    
    return detected_emotion, emotion_scores

def record_audio():
    print(f"Recording audio for {AUDIO_DURATION} seconds...")
    try:
        # Check if audio device is available
        devices = sd.query_devices()
        if len(devices) == 0:
            print("No audio devices found")
            # Create a silent audio file
            silent_audio = np.zeros(int(AUDIO_DURATION * AUDIO_SAMPLE_RATE), dtype='float32')
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav.write(temp_file.name, AUDIO_SAMPLE_RATE, (silent_audio * 32767).astype(np.int16))
            print("Created silent audio file as fallback")
            return temp_file.name
            
        # Record audio
        audio = sd.rec(int(AUDIO_DURATION * AUDIO_SAMPLE_RATE), 
                       samplerate=AUDIO_SAMPLE_RATE, 
                       channels=1, 
                       dtype='float32',
                       blocking=False)
        
        # Wait for recording to complete with timeout
        timeout = AUDIO_DURATION + 2  # Add 2 seconds buffer
        start_time = time.time()
        while sd.get_status().active and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        # If still recording after timeout, stop it
        if sd.get_status().active:
            sd.stop()
            print("Recording stopped due to timeout")
            
        print("Audio recording complete")
        
        # Create a temporary file to store the audio
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.write(temp_file.name, AUDIO_SAMPLE_RATE, (audio * 32767).astype(np.int16))
        return temp_file.name
        
    except Exception as e:
        print(f"Error recording audio: {e}")
        # Create a silent audio file as fallback
        silent_audio = np.zeros(int(AUDIO_DURATION * AUDIO_SAMPLE_RATE), dtype='float32')
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.write(temp_file.name, AUDIO_SAMPLE_RATE, (silent_audio * 32767).astype(np.int16))
        print("Created silent audio file as fallback")
        return temp_file.name

def gen_frames():
    global current_frame, is_recording
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return "Camera not available"
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process the frame to detect faces and emotions
        process_frame(frame)
        
        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', current_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in the format required by Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def analyze_emotion_data():
    """Analyze the recorded emotion data to extract meaningful insights"""
    if not recording_data["detected_emotions"]:
        return {
            "dominant_emotion": "Neutral",
            "distribution": {"Neutral": 1.0},
            "stability": 1.0,
            "changes": 0
        }
    
    # Get emotion distribution
    emotion_counts = Counter(recording_data["detected_emotions"])
    total_emotions = len(recording_data["detected_emotions"])
    emotion_distribution = {emotion: count/total_emotions for emotion, count in emotion_counts.items()}
    
    # Determine dominant emotion
    dominant_emotion = emotion_counts.most_common(1)[0][0]
    
    # Calculate emotional stability (how consistent the emotions were)
    # 1.0 means same emotion throughout, lower values mean more changes
    stability = max(emotion_distribution.values())
    
    # Count emotion changes
    changes = 0
    prev_emotion = None
    for emotion in recording_data["detected_emotions"]:
        if prev_emotion is not None and emotion != prev_emotion:
            changes += 1
        prev_emotion = emotion
    
    # Calculate change rate
    change_rate = changes / total_emotions if total_emotions > 0 else 0
    
    # Analyze average confidence scores for each emotion
    emotion_confidences = {}
    if recording_data["emotion_confidence_scores"]:
        avg_scores = np.mean(recording_data["emotion_confidence_scores"], axis=0)
        for i, label in enumerate(emotion_labels):
            emotion_confidences[label] = float(avg_scores[i])
    
    return {
        "dominant_emotion": dominant_emotion,
        "distribution": emotion_distribution,
        "stability": stability,
        "changes": changes,
        "change_rate": change_rate,
        "confidences": emotion_confidences
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, recording_data
    is_recording = True
    recording_data = {
        "frames": [],
        "detected_emotions": [],
        "emotion_confidence_scores": [],
        "emotion_timestamps": []
    }
    return jsonify({"status": "success", "message": "Recording started"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording
    is_recording = False
    
    try:
        print(f"Recording stopped. Detected {len(recording_data['detected_emotions'])} emotions.")
        
        # Analyze the recorded emotion data
        emotion_analysis = analyze_emotion_data()
        print(f"Emotion analysis: {emotion_analysis}")
        
        # Record audio and analyze
        audio_path = record_audio()
        
        try:
            voice_data = analyze_voice_tone(audio_path)
        except Exception as e:
            print(f"Error analyzing voice tone: {e}")
            # Provide default voice data
            voice_data = {
                "pitch": 150.0,
                "intensity": 0.05,
                "spectral_centroid": 2500.0,
                "spectral_bandwidth": 1500.0,
                "rms": 0.1,
                "zero_crossing_rate": 0.01,
                "tempo": 120.0
            }
        
        try:
            sentiment, transcribed_text, sentiment_score, subjectivity = analyze_speech_sentiment(audio_path)
        except Exception as e:
            print(f"Error in speech sentiment analysis: {e}")
            sentiment, transcribed_text, sentiment_score, subjectivity = "Neutral", "", 0.0, 0.5
        
        sentiment_data = {
            "sentiment": sentiment,
            "text": transcribed_text,
            "score": sentiment_score,
            "subjectivity": subjectivity
        }
        
        # Calculate comprehensive soft skills score
        try:
            scores = calculate_soft_skill_score(emotion_analysis, voice_data, sentiment_data)
        except Exception as e:
            print(f"Error calculating soft skills score: {e}")
            # Provide default scores
            scores = {
                "emotion_score": 7.0,
                "sentiment_score": 6.0,
                "pitch_score": 7.0,
                "intensity_score": 7.0,
                "voice_variability_score": 6.0,
                "tempo_score": 6.0,
                "final_score": 6.5
            }
        
        # Get detailed improvement suggestions
        try:
            suggestions = get_improvement_suggestions(scores["final_score"], emotion_analysis, voice_data, sentiment_data)
        except Exception as e:
            print(f"Error getting improvement suggestions: {e}")
            suggestions = "Practice maintaining consistent vocal tone and clear articulation. Try to express emotions more naturally in your facial expressions. Regular practice with feedback will help improve your communication skills."
        
        # Clean up temp file
        try:
            os.unlink(audio_path)
        except Exception as e:
            print(f"Error deleting temp file: {e}")
        
        # Return analysis results
        return jsonify({
            "status": "success",
            "analysis": {
                "emotion": emotion_analysis["dominant_emotion"],
                "pitch": float(voice_data["pitch"]),
                "intensity": float(voice_data["intensity"]),
                "sentiment": sentiment,
                "transcribed_text": transcribed_text,
                "scores": scores,
                "suggestions": suggestions,
                "emotion_distribution": emotion_analysis["distribution"],
                "emotion_stability": emotion_analysis["stability"]
            }
        })
    except Exception as e:
        print(f"Critical error in analysis: {e}")
        # Return a fallback analysis to prevent UI failure
        return jsonify({
            "status": "success",
            "analysis": {
                "emotion": "Neutral",
                "pitch": 150.0,
                "intensity": 0.05,
                "sentiment": "Neutral",
                "transcribed_text": "Speech analysis unavailable. Please try again.",
                "scores": {
                    "emotion_score": 7.0,
                    "sentiment_score": 6.0,
                    "pitch_score": 7.0,
                    "intensity_score": 7.0,
                    "voice_variability_score": 6.0,
                    "tempo_score": 6.0,
                    "final_score": 6.5
                },
                "suggestions": "Our analysis system encountered an issue. Please try again with clearer speech and good lighting for better results.",
                "emotion_distribution": {"Neutral": 1.0},
                "emotion_stability": 1.0
            }
        })

if __name__ == '__main__':
    app.run(debug=True, port=1250) 