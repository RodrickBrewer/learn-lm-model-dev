from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).cuda()

text = "A GPU is a processor."
enc = tokenizer(text, return_tensors="pt").to("cuda")

# Forward pass
with torch.no_grad():
    outputs = model(**enc)
    logits = outputs.logits  # (1, seq_len, vocab_size)

# Shift logits and labels for next-token prediction
shifted_logits = logits[:, :-1, :]      # remove last token prediction
shifted_labels = enc.input_ids[:, 1:]   # remove first token (target is next token)

# Compute cross-entropy loss manually
loss = F.cross_entropy(
    shifted_logits.reshape(-1, shifted_logits.size(-1)),
    shifted_labels.reshape(-1),
    # reduction="none"  # To see the loss per token
)

print("Cross-entropy loss:", loss.item())
# print("Cross-entropy loss:", loss, sep="\n")