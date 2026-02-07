# 🚀 SpeechInSight: Full ASR Pipeline

A complete end-to-end Speech Recognition system that segments long audio files (like meetings) and transcribes them using a custom-trained CRNN model.

## 🌟 Features
- **Smart Segmentation:** Uses VAD (Voice Activity Detection) to chop long audio into sentence-level clips.
- **Custom ASR Model:** A PyTorch-based CRNN (CNN + GRU + CTC) trained on LibriSpeech.
- **Interactive Dashboard:** Built with Streamlit for easy drag-and-drop usage.
- **Data Export:** auto-generates CSV transcripts and ZIP archives of audio segments.

## 🛠️ Installation

1. Clone the repo:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/SpeechInSight-Segmenter.git](https://github.com/YOUR_USERNAME/SpeechInSight-Segmenter.git)
   cd SpeechInSight-Segmenter
2. **Download the Model:**
   > ⚠️ **Note:** The model weights (`best_model.pth`) are not included in this repo due to size limits.
   > Please download the model from (https://drive.google.com/drive/folders/1Grk9U1PUF_YknbdXqUxHF5vpkfPvvl4C?usp=sharing) and place it in the root folder.
