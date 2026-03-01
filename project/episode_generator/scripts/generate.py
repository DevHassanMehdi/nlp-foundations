from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

import torch

from .episode_formatter import compose_episode_from_plan
from .episode_planner import create_episode_plan, section_prompts
from .ngram_model import BigramLM, load_model
from .pretrained_model import (
    PretrainedGenerator,
    generate_with_pretrained,
    load_pretrained_generator,
)
from .word_lstm import (
    WordLSTM,
    WordVocab,
    build_model_from_checkpoint,
    generate as generate_word_lstm,
)

BASE_DIR = Path(__file__).resolve().parents[1]


def _clean_section(text: str) -> str:
    text = " ".join((text or "").split()).strip()
    if not text:
        return ""
    words = [w for w in text.lower().split() if w.isalpha()]
    if words:
        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio < 0.38:
            return ""
    if text.count(",") > max(4, len(text) // 30):
        return ""
    if ".." in text or ",,." in text or ".,,." in text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text


def _strip_prompt_echo(generated: str, prompt: str) -> str:
    g = " ".join(generated.split()).strip()
    p = " ".join(prompt.split()).strip()
    if not g or not p:
        return g
    low_g = g.lower()
    low_p = p.lower()
    if low_g.startswith(low_p):
        tail = g[len(p) :].lstrip(" ,.;:-")
        return tail or g
    return g


def _generate_sections(
    gen_fn: Callable[[str, int, float, int], str],
    prompts: Dict[str, str],
    base_context: str,
    temp: float,
    seed: int,
) -> Dict[str, str]:
    sections: Dict[str, str] = {}

    budget = {
        "logline": 42,
        "setup": 24,
        "inciting_incident": 24,
        "rising_action": 44,
        "climax": 24,
        "resolution": 24,
        "scene_1": 20,
        "scene_2": 20,
        "scene_3": 20,
        "scene_4": 20,
        "dialogue_a": 14,
        "dialogue_b": 14,
        "dialogue_c": 14,
    }

    for i, key in enumerate(prompts.keys()):
        p = prompts[key]
        combined_prompt = f"{base_context} {p}"
        raw = gen_fn(combined_prompt, budget[key], temp, seed + i * 7)
        raw = _strip_prompt_echo(raw, combined_prompt)
        sections[key] = _clean_section(raw)
    return sections


def generate_ngram_structured_with_model(
    model: BigramLM,
    prompt: str,
    temperature: float,
    seed: int,
) -> str:
    plan = create_episode_plan(prompt)
    s_prompts = section_prompts(plan)

    def _gen(p: str, max_words: int, _: float, s: int) -> str:
        return model.generate(p, max_words=max_words, seed=s)

    base_context = ""
    sections = _generate_sections(_gen, s_prompts, base_context, temperature, seed)
    return compose_episode_from_plan(plan, sections, "N-gram (word-level)")


def generate_lstm_structured_with_model(
    model: WordLSTM,
    vocab: WordVocab,
    prompt: str,
    temperature: float,
    seed: int,
) -> str:
    plan = create_episode_plan(prompt)
    s_prompts = section_prompts(plan)

    def _gen(p: str, max_words: int, temp: float, s: int) -> str:
        return generate_word_lstm(
            model,
            vocab,
            prompt=p.lower(),
            max_words=max_words,
            temperature=temp,
            top_k=16,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            seed=s,
        )

    base_context = ""
    sections = _generate_sections(_gen, s_prompts, base_context, temperature, seed)
    return compose_episode_from_plan(plan, sections, "Word-LSTM (word-level)")


def generate_pretrained_structured_with_model(
    generator: PretrainedGenerator,
    prompt: str,
    temperature: float,
    seed: int,
) -> str:
    plan = create_episode_plan(prompt)
    s_prompts = section_prompts(plan)

    def _gen(p: str, max_words: int, temp: float, s: int) -> str:
        instruction = (
            "Write one coherent TV episode section in plain English. "
            "Keep character names and events consistent with the prompt. "
            f"Instruction: {p}"
        )
        return generate_with_pretrained(
            generator,
            prompt=instruction,
            max_words=max_words,
            temperature=temp,
            seed=s,
        )

    base_context = ""
    sections = _generate_sections(_gen, s_prompts, base_context, temperature, seed)
    model_label = generator.model_name.split("/")[-1]
    return compose_episode_from_plan(plan, sections, f"Transformer LM ({model_label})")


def generate_ngram_structured(prompt: str, temperature: float, seed: int) -> str:
    model = load_model(BASE_DIR / "models" / "ngram_model.json")
    return generate_ngram_structured_with_model(model, prompt, temperature, seed)


def generate_lstm_structured(prompt: str, temperature: float, seed: int) -> str:
    checkpoint = torch.load(BASE_DIR / "models" / "word_lstm.pt", map_location="cpu")
    vocab = WordVocab(stoi={t: i for i, t in enumerate(checkpoint["vocab"])}, itos=checkpoint["vocab"])
    model = build_model_from_checkpoint(checkpoint, vocab_size=len(vocab.itos))
    model.load_state_dict(checkpoint["state_dict"])
    return generate_lstm_structured_with_model(model, vocab, prompt, temperature, seed)


def generate_pretrained_structured(prompt: str, temperature: float, seed: int) -> str:
    generator = load_pretrained_generator()
    return generate_pretrained_structured_with_model(generator, prompt, temperature, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structured episode text")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ngram_text = generate_ngram_structured(args.prompt, args.temperature, args.seed)
    lstm_text = generate_lstm_structured(args.prompt, args.temperature, args.seed)
    pretrained_text = generate_pretrained_structured(args.prompt, args.temperature, args.seed)

    print("\n=== N-gram Structured Episode ===\n")
    print(ngram_text)
    print("\n=== Word-LSTM Structured Episode ===\n")
    print(lstm_text)
    print("\n=== Transformer LM Structured Episode ===\n")
    print(pretrained_text)


if __name__ == "__main__":
    main()
