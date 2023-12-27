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


# Function to reduce power of one speaker in the voice of another
def reduce_power(speaker_0_audio, speaker_1_audio, reduction_factor=0.5):
    # Ensure both audio segments have the same length
    min_length = min(len(speaker_0_audio), len(speaker_1_audio))
    speaker_0_audio = speaker_0_audio[:min_length]
    speaker_1_audio = speaker_1_audio[:min_length]

    # Apply gain reduction to Speaker 1 audio
    reduced_speaker_1 = speaker_1_audio.apply_gain(-20 * reduction_factor)  # -20 * log10(reduction_factor)

    # Combine the audio of both speakers
    final_audio = speaker_0_audio + reduced_speaker_1

    return final_audio
# Load the original audio file
audio = AudioSegment.from_file(audio_file)


# Extract audio for speaker 00
speaker_00_audio = extract_speaker_audio(diarization, "SPEAKER_00")
speaker_00_audio.export("./results/speaker_00_audio.mp3", format="mp3")

# Extract audio for speaker 01
speaker_01_audio = extract_speaker_audio(diarization, "SPEAKER_01")
speaker_01_audio.export("./results/speaker_01_audio.mp3", format="mp3")

# Reduce power of Speaker 1 in the voice of Speaker 0
final_audio = reduce_power(speaker_00_audio, speaker_01_audio, reduction_factor=0.5)

# Export the final audio
final_audio.export("final_audio.mp3", format="mp3")