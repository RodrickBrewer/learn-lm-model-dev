import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16).cuda()

# Load your LoRA adapter
model = PeftModel.from_pretrained(model, "tiny_sft_lora")

# Merge LoRA with base weights for faster inference
model = model.merge_and_unload()

prompt = "Terron is"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        # temperature=0.8
    )

print("\n=== Model Output ===")
print(tokenizer.decode(out[0], skip_special_tokens=True))
