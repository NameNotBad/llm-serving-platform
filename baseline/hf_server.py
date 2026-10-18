
import time
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

app = FastAPI(title="LLM Serving Baseline")

# Load model once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
).to("cuda").eval()

# Naive baseline: serialize GPU generation
generation_lock = Lock()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    max_new_tokens: int = Field(default=128, ge=1, le=128)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/generate")
def generate(request: GenerateRequest):
    lock_wait_start = time.perf_counter()

    with generation_lock:
        queue_ms = (time.perf_counter() - lock_wait_start) * 1000
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": request.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")

        input_tokens = inputs["input_ids"].shape[1]

        if input_tokens > 2048:
            raise HTTPException(
                status_code=422,
                detail="Input exceeds 2048 tokens",
            )

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        generated = outputs[0, input_tokens:]
        output_tokens = generated.shape[0]

        response = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

    return {
        "response": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generation_ms": round(elapsed * 1000, 2),
        "tokens_per_second": round(output_tokens / elapsed, 2),
        "queue_ms": round(queue_ms, 2),
    }
