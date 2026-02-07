import os
import shutil
import torch
import torchaudio


def segment_and_save(input_file, output_folder="segmented_clips"):
    """
    Uses Silero VAD (via Torch Hub) to detect speech and split audio.
    No HuggingFace tokens or dependency hell required.
    """

    # 1. Setup Folders
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # 2. Load Silero VAD Model
    print("⏳ Loading Silero VAD model...")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  trust_repo=True)

    (get_speech_timestamps, _, read_audio, _, _) = utils

    # 3. Read Audio
    print(f"🕵️ Analyzing '{input_file}'...")
    wav = read_audio(input_file, sampling_rate=16000)

    # 4. Get Speech Timestamps
    # min_speech_duration_ms: Ignore clicks/pops shorter than 250ms
    # min_silence_duration_ms: Join words if silence is less than 100ms
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100
    )

    # 5. Load Original Audio for Slicing (High Quality)
    # We load again with torchaudio to ensure we save in original quality
    waveform, sample_rate = torchaudio.load(input_file)

    saved_files = []
    print(f"✂️ Found {len(speech_timestamps)} segments. Slicing...")

    for i, timestamp in enumerate(speech_timestamps):
        # Convert Silero timestamps (samples at 16k) to seconds
        start_sec = timestamp['start'] / 16000
        end_sec = timestamp['end'] / 16000

        # Convert seconds to Original Sample Rate indices
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)

        # Slice
        segment = waveform[:, start_sample:end_sample]

        # Save
        filename = f"seg_{i:03d}.wav"
        save_path = os.path.join(output_folder, filename)
        torchaudio.save(save_path, segment, sample_rate)
        saved_files.append(save_path)

    return saved_files
