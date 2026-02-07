import streamlit as st
import os
import shutil
import time
import pandas as pd
from segmentation import segment_and_save
from model import load_model, transcribe_clip

# --- SETUP ---
st.set_page_config(page_title="SpeechInSight Pipeline", layout="wide")
st.title("🚀 SpeechInSight: Full Pipeline")

# Load Model Once (Cache it for speed)
@st.cache_resource
def get_asr_model():
    return load_model("best_model.pth")

model, device = get_asr_model()

if model is None:
    st.error("❌ 'best_model.pth' not found! Please download it from Drive and put it in this folder.")
    st.stop()

# --- UI ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Input")
    uploaded_file = st.file_uploader("Upload Audio/Video", type=["wav", "mp3", "mp4", "flac"])

if uploaded_file:
    # Save Upload
    input_path = "input_temp" + os.path.splitext(uploaded_file.name)[1]
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with col1:
        st.success("File Uploaded!")
        if st.button("▶️ Run Pipeline"):
            
            output_folder = "pipeline_output"
            progress = st.progress(0)
            status = st.empty()

            # --- STEP 1: SEGMENTATION ---
            status.text("✂️ Step 1: Segmentation (Cutting audio)...")
            clips = segment_and_save(input_path, output_folder)
            progress.progress(40)

            # --- STEP 2: TRANSCRIPTION ---
            status.text("🧠 Step 2: Transcribing clips with CRNN...")
            
            results = []
            for i, clip_path in enumerate(clips):
                # Transcribe
                text = transcribe_clip(clip_path, model, device)
                
                # Store Data
                results.append({
                    "Filename": os.path.basename(clip_path),
                    "Transcript": text,
                    "Path": clip_path
                })
                # Update progress bar slightly for each file
                current_prog = 40 + int((i / len(clips)) * 60)
                progress.progress(current_prog)
            
            progress.progress(100)
            status.success("✅ Pipeline Complete!")

            # --- DISPLAY RESULTS ---
            with col2:
                st.subheader("2. Results")
                
                # 1. Show Data Table
                df = pd.DataFrame(results)
                st.dataframe(df[["Filename", "Transcript"]])

                # 2. Export ZIP (For Teammate - Audio Only)
                shutil.make_archive("audio_clips_for_teammate", 'zip', output_folder)
                with open("audio_clips_for_teammate.zip", "rb") as fp:
                    st.download_button("📦 Download Audio Clips (For Teammate)", fp, "clips.zip")
                
                # 3. Export CSV (For You - Text Emotion)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📄 Download Transcripts (For Emotion AI)", csv, "transcripts.csv", "text/csv")
                
                st.divider()
                st.write("🎧 **Listen to Segments:**")
                for item in results:
                    st.markdown(f"**{item['Transcript']}**")
                    st.audio(item['Path'])