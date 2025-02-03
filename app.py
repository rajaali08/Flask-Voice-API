from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import librosa
import io

app = Flask(__name__)

# ✅ Fix CORS: Allow frontend access from any origin (or restrict to your domain)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Load trained SVM model
model = joblib.load("decision_tree.pkl")  # Ensure this file exists

# Load scaler if used
try:
    scaler = joblib.load("scaler.pkl")  # If you saved a scaler
except:
    scaler = None

# Function to extract features from an audio file
def extract_features(audio_data, sr):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr)

    # Taking mean values to reduce dimensionality
    mfccs_mean = np.mean(mfccs, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    mel_mean = np.mean(mel, axis=1)

    return np.hstack([mfccs_mean, chroma_mean, mel_mean])

@app.route("/")
def home():
    return "Voice Pathology Detection API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the uploaded audio file
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        audio, sr = librosa.load(io.BytesIO(file.read()), sr=22050)

        # Extract features
        features = extract_features(audio, sr).reshape(1, -1)

        # Scale features if scaler exists
        if scaler:
            features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features)[0]
        result = "Pathological Voice" if prediction == 1 else "Healthy Voice"

        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
