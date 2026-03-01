from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL_NAMES = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "distilgpt2",
)


@dataclass
class PretrainedGenerator:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device
    model_name: str


def load_pretrained_generator(model_name: str = DEFAULT_MODEL_NAME) -> PretrainedGenerator:
    """Load a local GPT-style model with fallback tiers for lower-memory machines."""
    requested = [model_name] + [m for m in FALLBACK_MODEL_NAMES if m != model_name]
    last_error: Exception | None = None

    for candidate in requested:
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device.type == "cuda" else torch.float32

            model = AutoModelForCausalLM.from_pretrained(candidate, torch_dtype=dtype)
            model.to(device)
            model.eval()
            return PretrainedGenerator(
                tokenizer=tokenizer,
                model=model,
                device=device,
                model_name=candidate,
            )
        except Exception as exc:  # pragma: no cover - runtime fallback path
            last_error = exc
            continue

    if last_error is not None:
        raise RuntimeError("Could not load any pretrained fallback model") from last_error
    raise RuntimeError("Could not load pretrained model")


def generate_with_pretrained(
    generator: PretrainedGenerator,
    prompt: str,
    max_words: int = 64,
    temperature: float = 0.7,
    seed: Optional[int] = None,
) -> str:
    """Generate one short coherent section from an instruction prompt."""
    if seed is not None:
        set_seed(seed)

    max_new_tokens = max(24, min(180, int(max_words * 2.1)))
    temperature = max(0.15, min(temperature, 1.0))

    if hasattr(generator.tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        encoded = generator.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(encoded, "to"):
            encoded = encoded.to(generator.device)

        # HF tokenizers can return either a tensor or a BatchEncoding here.
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded
            attention_mask = torch.ones_like(input_ids, device=generator.device)
        else:
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=generator.device)
    else:
        encoded_obj = generator.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(generator.device)
        input_ids = encoded_obj["input_ids"]
        attention_mask = encoded_obj.get("attention_mask")

    with torch.no_grad():
        output_ids = generator.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.06,
            no_repeat_ngram_size=3,
            pad_token_id=generator.tokenizer.pad_token_id,
            eos_token_id=generator.tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][input_ids.shape[-1] :]
    text = generator.tokenizer.decode(new_ids, skip_special_tokens=True)
    text = " ".join(text.split()).strip()
    text = _strip_instruction_leak(text)
    return text


def _strip_instruction_leak(text: str) -> str:
    if not text:
        return ""

    # Remove common instruction boilerplate if the model echoes it.
    leak_markers = [
        "you are writing a tv episode draft",
        "write exactly one section",
        "keep character names and events consistent",
        "avoid bullets unless",
        "section instruction:",
        "section text:",
    ]
    lowered = text.lower()
    for marker in leak_markers:
        idx = lowered.find(marker)
        if idx != -1 and idx < 120:
            tail = text[idx:]
            if "." in tail:
                text = tail.split(".", 1)[-1].strip()
                lowered = text.lower()

    text = re.sub(r"^(section\s*(instruction|text)\s*:)\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip()
