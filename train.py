import torch
from datasets import Audio, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MusicgenForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# quantification for a 8GB VRAM GPU (probably not required for newer cards)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# loading model
model_id = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained(model_id)
model = MusicgenForConditionalGeneration.from_pretrained(
    model_id, quantization_config=bnb_config, device_map={"": 0}
)

# setup Lora
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

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


# map dataset to smaller batch to save ram
tokenized_dataset = dataset.map(preprocess_function, batched=True, batch_size=1)

# training
training_args = TrainingArguments(
    output_dir="./musicgen-sewerslvt-lora",
    per_device_train_batch_size=1,  # Keep this at 1 for 8GB
    gradient_accumulation_steps=8,  # Simulates a batch size of 8
    warmup_steps=10,
    max_steps=100,
    learning_rate=2e-4,
    fp16=True,  # Use half-precision
    logging_steps=1,
    optim="adamw_bnb_8bit",  # Use 8-bit optimizer
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
