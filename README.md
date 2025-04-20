# Welcome to the Grammar Scoring Engine 🎙️📊

A machine learning model that analyzes spoken audio to predict grammar quality using acoustic features (MFCCs, Spectral Contrast, ZCR) and provides real-time feedback through an interactive web interface.
Here is the screenshot:- 
![image](https://github.com/user-attachments/assets/b8a899dc-4f19-4cfe-bbeb-31a40fff29bc)

## Features ✨

- **Audio Analysis**: Extracts 20+ acoustic features including:
  - MFCCs (Mel-frequency cepstral coefficients)
  - Spectral Contrast
  - Zero Crossing Rate (ZCR)
- **Machine Learning**: Random Forest Regressor trained on speech patterns
- **Real-time Feedback**: Instant grammar score prediction (0.0-1.0 scale)
- **Visual Analytics**:
  - Interactive waveform display
  - MFCC heatmap visualization
- **File Support**: Processes WAV and MP3 audio formats

## Tech Stack 🛠️

**Core Technologies**
- `Librosa` (Audio feature extraction)
- `scikit-learn` (Machine Learning pipeline)
- `Streamlit` (Web interface)
- `NumPy`/`Pandas` (Data processing)

**Supporting Libraries**
- Matplotlib (Visualizations)
- Joblib (Model persistence)
- LanguageTool-Python (Grammar validation)

**Dataset 📚**
- Using Mozilla's Common Voice dataset from Kaggle:
- Common Voice Dataset contains 50,000+ validated speech samples
- Annotated with speaker metadata and text transcripts

## Acknowledgments 🙏
- Mozilla Common Voice team for speech dataset
- Librosa community for audio processing tools
- Streamlit for interactive web components
