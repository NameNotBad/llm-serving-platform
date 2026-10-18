import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
).to("cuda")

model.eval()

messages = [
    {
        "role": "user",
        "content": "Explain what a KV cache is in one short paragraph."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(text, return_tensors="pt").to("cuda")

torch.cuda.synchronize()
start = time.perf_counter()

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)

print("\n=== Response ===")
print(response)

print("\n=== Metrics ===")
print(f"Input tokens: {inputs['input_ids'].shape[1]}")
print(f"Output tokens: {generated_tokens}")
print(f"Generation time: {elapsed:.3f} s")
print(f"Throughput: {generated_tokens / elapsed:.2f} tokens/s")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
