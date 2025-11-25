from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).cuda()

text = "A GPU is a processor."
enc = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**enc)
    logits = outputs.logits

# Next token should be:
# "GPU" -> "is" -> "a" -> "processor" -> "."
# Let's inspect the probabilities for each step

for i in range(enc.input_ids.shape[1] - 1):
    current_token = enc.input_ids[0, i].item()
    target_token = enc.input_ids[0, i+1].item()
    
    logit_vec = logits[0, i]  # logits for next-token prediction at step i
    probs = torch.softmax(logit_vec, dim=-1)
    
    p_correct = probs[target_token].item()
    print(f"Step {i}: P(correct) = {p_correct:.6f}")