import streamlit as st
import librosa
import numpy as np
import joblib

# Load saved model, scaler, and encoder
model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Feature extraction function
def extract_features(y, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack((mfcc, chroma, mel))

# Streamlit UI
st.title("🎙️ Speech Emotion Classifier")
st.markdown("Upload a speech/audio file to predict the emotion.")

audio_file = st.file_uploader("Choose a .wav file", type=["wav"])

if audio_file is not None:
    y, sr = librosa.load(audio_file, duration=3, offset=0.5)
    features = extract_features(y, sr)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    emotion = le.inverse_transform(prediction)[0]

    st.success(f"🎧 Predicted Emotion: **{emotion.upper()}**")
