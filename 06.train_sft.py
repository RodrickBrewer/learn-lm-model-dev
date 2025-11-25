import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16).cuda()

# Load your dataset
dataset = load_dataset("json", data_files="06.dataset.jsonl")

# Format dataset for causal LM
def preprocess(example):
    # text = example["input"] + " " + example["output"]
    text = f"<input> {example['input']} <output> {example['output']}"
    tokens = tokenizer(text, truncation=True)  #!~
    return tokens

dataset = dataset.map(preprocess)

# Setup LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./sft_out",
    per_device_train_batch_size=1,
    num_train_epochs=10,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# Save LoRA weights
model.save_pretrained("tiny_sft_lora")
