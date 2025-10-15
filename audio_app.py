from flask import Flask, render_template, Response
import threading
import time
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import openl3

# Ensure TensorFlow uses CPU
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)
audio_events = []

# YAMNet setup
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)

# OpenL3 setup
openl3_model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="env")

def capture_audio(duration=1, sample_rate=16000):
    """Captures audio for a specified duration."""
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio_data.flatten()

def preprocess_audio(audio_data, sample_rate=16000):
    """Preprocesses the audio data for the models."""
    if sample_rate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
    return audio_data

def analyze_yamnet(audio_data, sample_rate=16000):
    """Analyzes audio using YAMNet."""
    scores, embeddings, spectrogram = yamnet_model(audio_data)
    class_map_path = hub.resolve(yamnet_model_handle) + '/assets/yamnet_class_map.csv'
    try:
        with tf.io.gfile.GFile(class_map_path) as f:
            class_names = [name.rstrip('\n') for name in f.readlines()]
        inference = scores.numpy().mean(axis=0)
        top_class = class_names[inference.argmax()]
        return top_class
    except Exception as e:
        print(f"Error loading class names: {e}")
        return "Unknown sound"

def analyze_openl3(audio_data, sample_rate=16000):
    """Analyzes audio using OpenL3."""
    embeddings, timestamps = openl3.get_audio_embedding(audio_data, sample_rate, model=openl3_model)
    # For simplicity, we'll return the mean embedding
    mean_embedding = np.mean(embeddings, axis=0)
    return mean_embedding.tolist() # Convert to list for JSON serialization

def audio_processing_loop():
    """Continuously captures and analyzes audio."""
    while True:
        audio = capture_audio()
        processed_audio = preprocess_audio(audio)
        yamnet_result = analyze_yamnet(processed_audio)
        openl3_result = analyze_openl3(processed_audio)
        audio_events.append({"yamnet": yamnet_result, "openl3": openl3_result})
        time.sleep(1) # Adjust as needed.

@app.route('/')
def index():
    return render_template('indexxx.html')

def generate():
    """Generator function that yields events."""
    global audio_events
    while True:
        if len(audio_events) > 0:
            event = audio_events.pop(0)
            yield f"data: {event}\n\n"
        time.sleep(0.1)

@app.route('/stream')
def stream():
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    threading.Thread(target=audio_processing_loop).start()
    app.run(debug=True)