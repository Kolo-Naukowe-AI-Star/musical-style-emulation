import scipy.io.wavfile
import torch
from peft import PeftModel
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# 1. Setup paths and device
model_id = "facebook/musicgen-small"
adapter_path = "./musicgen-style-lora"  # Path to your 'output_dir' from training
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Base Model and Processor
processor = AutoProcessor.from_pretrained(model_id)
base_model = MusicgenForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16
).to(device)

# 3. Load your fine-tuned LoRA weights
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval() # Set to evaluation mode

# 4. Generate Audio
prompt = "A high-energy electronic track in the style of Sewerslvt"
inputs = processor(text=[prompt], return_tensors="pt").to(device)

print("Generating audio...")
with torch.no_grad():
    # max_new_tokens=256 gives ~5 seconds of audio for a quick test
    audio_tokens = model.generate(**inputs, max_new_tokens=256)

# 5. Save to .wav
sampling_rate = model.config.audio_encoder.sampling_rate
# Remove batch and channel dims for saving: [1, 1, samples] -> [samples]
audio_data = audio_tokens[0, 0].cpu().numpy()

scipy.io.wavfile.write("output_test.wav", rate=sampling_rate, data=audio_data)
print("Saved to output_test.wav")
