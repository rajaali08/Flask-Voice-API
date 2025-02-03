import os
import io
import numpy as np
import librosa
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# ✅ Load trained model
try:
    model = joblib.load("decision_tree.pkl")  # Ensure this file exists
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Prevent crashes if model is missing

# ✅ Load scaler if available
try:
    scaler = joblib.load("scaler.pkl")
    print("✅ Scaler loaded successfully!")
except:
    scaler = None  # Continue even if scaler is missing

# ✅ Feature extraction function
def extract_features(audio_data, sr):
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio_data, sr=sr)

        # Taking mean values to reduce dimensionality
        mfccs_mean = np.mean(mfccs, axis=1)
        chroma_mean = np.mean(chroma, axis=1)
        mel_mean = np.mean(mel, axis=1)

        return np.hstack([mfccs_mean, chroma_mean, mel_mean])
    
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None

# ✅ Root route (for testing)
@app.route("/")
def home():
    return jsonify({"message": "🚀 Voice Pathology Detection API is running!"})

# ✅ Prediction route (NO CORS, DIRECT ANALYSIS)
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please check your deployment."}), 500

    try:
        # ✅ Get the uploaded audio file
        if "file" not in request.files:
            return jsonify({"error": "❌ No file uploaded"}), 400

        file = request.files["file"]
        audio, sr = librosa.load(io.BytesIO(file.read()), sr=22050)

        # ✅ Extract features
        features = extract_features(audio, sr)
        if features is None:
            return jsonify({"error": "❌ Feature extraction failed."}), 500

        features = features.reshape(1, -1)

        # ✅ Scale features if scaler exists
        if scaler:
            features = scaler.transform(features)

        # ✅ Make prediction
        prediction = model.predict(features)[0]
        result = "⚠️ Pathological Voice Detected!" if prediction == 1 else "✅ Healthy Voice"

        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": f"🔥 Error processing request: {str(e)}"}), 500

# ✅ Ensure the app runs on a proper port (important for deployment)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
