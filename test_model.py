import librosa
import numpy as np
import joblib

# Load saved model and preprocessing tools
model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Feature extraction
def extract_features(y, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack((mfcc, chroma, mel))

# Emotion prediction function
def predict_emotion(audio_path):
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)
    features = extract_features(y, sr)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    emotion = le.inverse_transform(prediction)[0]
    return emotion

# Example usage
if __name__ == "__main__":
    path = input("Enter path to audio file (WAV format): ")
    emotion = predict_emotion(path)
    print(f"Predicted Emotion: {emotion}")
