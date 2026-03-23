import torch
from datasets import Audio, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          MusicgenForConditionalGeneration, Trainer,
                          TrainingArguments)

# # quantification for a 8GB VRAM GPU (probably not required for newer cards)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

# loading model
model_id = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained(model_id)
model = MusicgenForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto",
)

# setup Lora
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# load dataset
dataset = load_dataset("json", data_files="data.jsonl", split="train")
dataset = dataset.cast_column("path", Audio(sampling_rate=32000))


# preprocessing (resample to 32000)
def preprocess_function(examples):
    audio = [x["array"] for x in examples["path"]]
    inputs = processor(
        audio=audio,
        sampling_rate=32000,
        text=examples["description"],
        padding=True,
        return_tensors="pt",
    )
    return inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True, batch_size=1)

# training
training_args = TrainingArguments(
    output_dir="./musicgen-style-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_total_limit=2,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
