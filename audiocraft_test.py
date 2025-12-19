import torchaudio

from audiocraft.data.audio import audio_write
from audiocraft.models.musicgen import MusicGen

model = MusicGen.get_pretrained("facebook/musicgen-melody")
model.set_generation_params(duration=8)  # generate 8 seconds.
wav = model.generate_unconditional(4)  # generates 4 unconditional audio samples
descriptions = ["happy rock", "energetic EDM", "sad jazz"]
wav = model.generate(descriptions)  # generates 3 samples.

melody, sr = torchaudio.load("./assets/bach.mp3")
# generates using the melody from the given audio and the provided descriptions.
wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(
        f"{idx}",
        one_wav.cpu(),
        model.sample_rate,
        strategy="loudness",
        loudness_compressor=True,
    )
