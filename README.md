# Diarization


Speaker diarization is the process of automatically identifying and segmenting an audio recording into distinct speech segments, where each segment corresponds to a particular speaker. In simpler words, the goal is to answer the question: who spoke when? It involves analyzing the audio signal to detect changes in speaker identity, and then grouping together segments that belong to the same speaker.

Speaker diarization is a key component of conversation analysis tools and can often be coupled with Automatic Speech Recognition (ASR) or Speech Emotion Recognition (SER) to extract meaningful information from conversational content. Hence, speaker diarization provides important information when performing speech analysis that involves several speakers 

## Install
* Torch 
```python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
* requirements
```python
pip install -r requirements.txt
```

# Acknowledgments

I would like to express my gratitude to the developers and contributors of pyannote-audio for providing an exceptional tool for audio diarization. Their work has greatly contributed to the success of this project.

Special thanks to the open-source community for their invaluable support and collaboration.

Address: pyannote-audio, GitHub Repository, https://github.com/pyannote/pyannote-audio.
