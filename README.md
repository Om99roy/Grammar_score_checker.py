# Welcome to the Grammar Scoring Engine 🎙️📊

A machine learning model that analyzes spoken audio to predict grammar quality using acoustic features (MFCCs, Spectral Contrast, ZCR) and provides real-time feedback through an interactive web interface.
Here is the screenshot:- 
![image](https://github.com/user-attachments/assets/b8a899dc-4f19-4cfe-bbeb-31a40fff29bc)

![image](https://github.com/user-attachments/assets/957bea4a-486b-4e33-a196-94417b2cf150)


**Demo video:** - [Grammar Scoring Engine Demo ▶️ ](https://youtu.be/lu1UdPp1ujI?si=unGBN42zbcJBbIxm)


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
  - As well as Download Full csv report
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
- Annotated with speaker metadata and text transcripts
- Used a curated subset of Mozilla's Common Voice dataset:
- **Samples**: 25,403 voice recordings (`cv-invalid.csv`)
- **Features**:  
  - Audio files (MP3 format)  
  - Text transcripts  
  - Speaker metadata (age, gender, accent)
  - **Souce**: [Common Voice on Kaggle](https://www.kaggle.com/datasets/mozillaorg/common-voice)
  - The full Common Voice dataset contains 2.5M+ recordings – see [official site](https://commonvoice.mozilla.org/).

## Acknowledgments 🙏
- Mozilla Common Voice team for speech dataset
- Librosa community for audio processing tools
- Streamlit for interactive web components
