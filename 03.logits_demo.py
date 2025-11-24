from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).cuda()

prompt = "A GPU is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)  # forward pass
    logits = outputs.logits  # shape: (batch=1, seq_len, vocab_size)

# Get logits for the last token in the prompt
last_token_logits = logits[0, -1]  # shape: (vocab_size,)

# Sort top 10 tokens by logit score
topk = torch.topk(last_token_logits, k=10)

print("\nTop 10 candidate next tokens:")
for score, token_id in zip(topk.values, topk.indices):
    tok = tokenizer.decode([token_id.item()])
    print(f"{score.item():>10.4f}  ->  {repr(tok)}")
