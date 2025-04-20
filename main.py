import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import librosa
import librosa.display
import tempfile
import streamlit as st
import io

# 1. ==================== Load the model ====================
model = joblib.load('grammar_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title("Welcome to the Grammar Scoring Engine 🧠✅ ")
uploaded_file = st.file_uploader("Upload your audio file 🔊", type=['wav', 'mp3'])

if uploaded_file:

    st.audio(uploaded_file, format='audio/wav')
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        audio_path = tmp.name

    # Load audio and extract features
    y, sr = librosa.load(audio_path, sr=22050)
    features = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
        features[f'mfcc_{i}_std'] = np.std(mfcc[i])

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(spectral_contrast.shape[0]):
        features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
        features[f'spectral_contrast_{i}_std'] = np.std(spectral_contrast[i])

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # 2. ==================== Predict the score ====================
    # Ensure correct feature order using scaler's feature names
    input_data = pd.DataFrame([features])[scaler.feature_names_in_]
    input_scaled = scaler.transform(input_data)
    score = model.predict(input_scaled)[0]
    st.metric("Grammar Score", f"{score:.2f}/7.0")

    # 3. ==================== Visualizations ====================
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    mfcc_img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax)
    cbar = fig.colorbar(mfcc_img, ax=ax, format="%+2.f")
    ax.set_title("MFCCs")
    st.pyplot(fig)

    # 4. ==================== Download Report ====================
    features['My_predicted_score'] = round(score, 4)
    report_df = pd.DataFrame([features])
    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label= "📥 Download Score Report (CSV)",
        data=csv_buffer.getvalue(),
        file_name="grammar_score_report.csv",
        mime="text/csv"
    )