import json
import os

import librosa  # You'll need: pip install librosa

audio_dir = "./data"
output_file = "data.jsonl"
artist_style = (
    "Atmospheric drum'n bass, breakcore, jungle, ethereal pads, harsh textures, in the style of Sewerslvt"
)

with open(output_file, "w") as f:
    for file in os.listdir(audio_dir):
        if file.endswith((".wav", ".mp3", ".flac")):
            path = os.path.normpath(os.path.join(audio_dir, file))

            # Get actual duration and BPM
            y, sr = librosa.load(path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            entry = {
                "path": path,
                "duration": duration,
                "sample_rate": sr,
                "description": artist_style,
                "genre": "Breakcore",
                "bpm": int(tempo),
            }
            f.write(json.dumps(entry) + "\n")
