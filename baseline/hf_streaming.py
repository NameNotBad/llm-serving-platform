import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


class TimedStreamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        self.token_times = []

    def put(self, value):
        if not self.next_tokens_are_prompt:
            self.token_times.append(time.perf_counter())
        super().put(value)


print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
).to("cuda").eval()

messages = [
    {
        "role": "user",
        "content": (
            "Explain how KV cache works in Transformer "
            "attention during autoregressive decoding."
        ),
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=False,
).to("cuda")

streamer = TimedStreamer(tokenizer)

torch.cuda.synchronize()
start = time.perf_counter()

print("\n=== Streaming Response ===")

with torch.inference_mode():
    model.generate(
        inputs,
        streamer=streamer,
        max_new_tokens=128,
        do_sample=False,
    )

torch.cuda.synchronize()
end = time.perf_counter()

times = streamer.token_times
n = len(times)

total_latency = end - start
ttft = times[0] - start if n else 0
tpot = (times[-1] - times[0]) / (n - 1) if n > 1 else 0

print("\n=== Metrics ===")
print(f"Input tokens: {inputs.shape[1]}")
print(f"Output tokens: {n}")
print(f"Total latency: {total_latency:.3f} s")
print(f"TTFT: {ttft * 1000:.2f} ms")
print(f"TPOT: {tpot * 1000:.2f} ms/token")
print(f"Throughput: {n / total_latency:.2f} tokens/s")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
