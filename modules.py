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


# Extract audio for speaker 00
speaker_00_audio = extract_speaker_audio(diarization, "SPEAKER_00")
speaker_00_audio.export("speaker_00_audio.mp3", format="mp3")

# Extract audio for speaker 01
speaker_01_audio = extract_speaker_audio(diarization, "SPEAKER_01")
speaker_01_audio.export("speaker_01_audio.mp3", format="mp3")