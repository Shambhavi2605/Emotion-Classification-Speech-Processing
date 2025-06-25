# Emotion-Classification-Speech-Processing
This project implements an end-to-end machine learning pipeline to classify emotions from speech audio using the RAVDESS dataset. It includes preprocessing, feature extraction (MFCC, Chroma, Mel Spectrogram), model training (LightGBM, XGBoost, SVM, etc.), evaluation, and deployment via a Streamlit web app.

# Objective

To build a robust emotion detection system that can classify speech into one of the following 8 emotional categories using audio processing and machine learning techniques.
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

# Dataset
We use the RAVDESS dataset (Ryerson Audio-Visual Database of Emotional Speech and Song), which contains audio recordings from 24 actors expressing various emotions. The dataset was parsed and labeled based on filename conventions.

# Preprocessing & Feature Extraction
Audio trimmed to 3 seconds, offset by 0.5s. Extracted features like mfccs, chroma, mel-spectrograms using librosa. Data augmentation techniques-pitch shift, time stretch, light gaussian noise and volume gain were added to training data.Label encoding and standardization applied on features.

# Models Tried
The following models were trained and evaluated:
- XGBoost
- Random Forest
- Logistic Regression
- SVM
- LightGBM (best-performing)

Final Model: LightGBM

- Accuracy: ~94%
- F1-Score: >90% for all classes

Meets evaluation criteria:
- Overall accuracy > 80%
- F1-score > 80%
- Class-wise accuracy > 75%

# Evaluation Metrics
- Confusion Matrix
- Classification Report
- Per-Class Accuracy
- Macro and Weighted F1 Scores

#Pipeline
- Load and label audio files from RAVDESS
- Extract features using librosa
- Apply data augmentation to training set
- Train ML models and evaluate
- Save trained LightGBM model using joblib
- Build and deploy frontend using Streamlit

#Testing
A Python script (test_model.py) allows testing the saved model by passing new audio files (WAV format). Also, the web app allows end-users to upload audio and receive emotion predictions.

#Web App (Streamlit)
The frontend is built using Streamlit. Users can upload an audio file, and the app returns the predicted emotion using the trained model.

Run it locally:

streamlit run streamlit_app.py

#Project Structure
.
├── Emotion_Classification.ipynb # Full development notebook
├── streamlit_app.py # Streamlit frontend code
├── test_model.py # Script to test saved model
├── metadata.csv # Parsed RAVDESS file metadata
├── saved_model.pkl # Trained LightGBM model
├── README.md # Project documentation
└── demo_video.mp4 # 2-minute demo video

📽#Demo
A short demo video (demo_video.mp4) demonstrates how the web app works by uploading sample speech audio.

#Requirements
Install dependencies:
- pip install -r requirements.txt
Main Libraries used:
- librosa
- scikit-learn
- xgboost, lightgbm
- pandas, numpy
- streamlit
- joblib

# Future Improvements
- Explore CNN/RNN models for temporal feature learning
- Use pre-trained speech embeddings like wav2vec2
- Improve UI/UX of the web app
- Add song classification functionality
