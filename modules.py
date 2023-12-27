import os
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment


audio_file = "test_files/arh_zahra.mp3"

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
diarization = pipeline(audio_file)

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

# create results directory
os.makedirs("./results", exist_ok=True)

# Function to extract speaker audio segments
def extract_speaker_audio(diarization, speaker_label):
    speaker_segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker == speaker_label:
            speaker_segments.append(turn)

    # Load the original audio file
    audio = AudioSegment.from_file(audio_file)

    # Extract speaker segments from the original audio
    speaker_audio = sum([audio[int(segment.start * 1000):int(segment.end * 1000)] for segment in speaker_segments])

    return speaker_audio


# Function to reduce volume of one speaker during overlap
def reduce_overlap_volume(speaker_1_audio, speaker_2_audio):
    overlap = speaker_1_audio.overlay(speaker_2_audio)

    # Reduce the volume of the second speaker during the overlap
    reduced_speaker_2 = overlap - speaker_2_audio

    # Remove the original volume of the second speaker during the overlap
    cleaned_audio = speaker_1_audio - overlap

    # Combine the reduced volume speaker and cleaned audio
    final_audio = cleaned_audio + reduced_speaker_2

    return final_audio

# Load the original audio file
audio = AudioSegment.from_file(audio_file)


# Extract audio for speaker 00
speaker_00_audio = extract_speaker_audio(diarization, "SPEAKER_00")
speaker_00_audio.export("./results/speaker_00_audio.mp3", format="mp3")

# Extract audio for speaker 01
speaker_01_audio = extract_speaker_audio(diarization, "SPEAKER_01")
speaker_01_audio.export("./results/speaker_01_audio.mp3", format="mp3")

# Reduce volume during overlap
final_audio = reduce_overlap_volume(speaker_00_audio, speaker_01_audio)

# Export the final audio
final_audio.export("results/final_audio.mp3", format="mp3")