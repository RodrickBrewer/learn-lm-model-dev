from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
model = model.cuda()

prompt = "What is 2+3?"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
    )
    
print("\nModel output:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
# print("\ntype of output", type(output))
# print(output)