
import random


from transformers import AutoTokenizer


def generate_token_length_prompts(
    count: int,
    target_tokens: int,
    seed: int = 42,
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> list[str]:
    if count < 1 or target_tokens < 1:
        raise ValueError("count and target_tokens must be positive")

    tokenizer = AutoTokenizer.from_pretrained(model)

    def chat_length(text: str) -> int:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        return len(encoded["input_ids"])

    prompts = []

    for request_id in range(count):
        candidate = generate_prompts(
            count=1,
            target_words=max(target_tokens * 2, 256),
            seed=seed + request_id,
        )[0]

        prefix = f"Benchmark request {request_id}: "
        words = candidate.split()

        low, high = 0, len(words)

        while low < high:
            mid = (low + high + 1) // 2
            text = prefix + " ".join(words[:mid])

            if chat_length(text) <= target_tokens:
                low = mid
            else:
                high = mid - 1

        prompt = prefix + " ".join(words[:low])

        # If we are short by 1–2 tokens, try a short suffix.
        suffixes = [
            "",
            " a",
            " the",
            " test",
            " example",
            " data",
            " performance",
            " system",
            " model",
            ".",
            "!",
        ]

        matched = None

        for suffix in suffixes:
            trial = prompt + suffix

            if chat_length(trial) == target_tokens:
                matched = trial
                break

        if matched is None:
            raise RuntimeError(
                f"Cannot generate exactly {target_tokens} "
                f"tokens for request {request_id}"
            )

        prompts.append(matched)

    return prompts


TOPICS = [
    "Transformer architecture",
    "GPU memory management",
    "distributed systems",
    "database indexing",
    "network protocols",
    "machine learning optimization",
    "operating system scheduling",
    "LLM inference serving",
    "software performance engineering",
    "cloud infrastructure",
]


def generate_prompts(
    count: int,
    target_words: int = 128,
    seed: int = 42,
) -> list[str]:
    if count < 1 or target_words < 1:
        raise ValueError("count and target_words must be positive")

    rng = random.Random(seed)
    prompts = []

    for request_id in range(count):
        topic = rng.choice(TOPICS)

        prefix = (
            f"Request ID: {request_id}. "
            f"Analyze the following technical notes about {topic}. "
            "Summarize the key ideas and explain their implications.\n"
        )

        vocabulary = [
            "compute", "memory", "latency", "throughput",
            "parallelism", "scheduling", "bandwidth",
            "optimization", "architecture", "performance",
            "inference", "deployment", "batching", "cache",
        ]

        words = [
            rng.choice(vocabulary)
            for _ in range(target_words)
        ]

        prompt = prefix + " ".join(words)
        prompts.append(prompt)

    return prompts


if __name__ == "__main__":
    prompts = generate_prompts(
        count=3,
        target_words=128,
        seed=42,
    )

    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i + 1} ---")
        print(prompt)
