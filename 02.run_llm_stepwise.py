from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).cuda()

prompt = "What is 2+3?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        temperature=0.7,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True
    )
    
# Extract the final sequence of token IDs
token_ids = outputs.sequences[0]

print("\ntype of outputs", type(outputs))
print(outputs)

print(f"\n=== Decoded tokens (one by one) len={len(token_ids)} ===")
for tok in token_ids: #[:15]:  # print first ~15 tokens only
    tid = tok.item()
    text = tokenizer.decode([tid], skip_special_tokens=False)
    print(f"{tid:>6}  ->  {repr(text)}")
