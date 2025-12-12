import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16).cuda()
model = PeftModel.from_pretrained(model, "tiny_sft_lora")
model = model.merge_and_unload()

prompt = "<input> Describe Terron. <output>"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False    # IMPORTANT: we want deterministic testing
    )

print("\n=== Model Output ===")
print(tokenizer.decode(out[0], skip_special_tokens=True))