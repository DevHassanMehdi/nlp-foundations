from __future__ import annotations

import re
from typing import Dict, List


def _extract_characters(prompt: str) -> List[str]:
    names = re.findall(r"\b[A-Z][a-z]{2,}\b", prompt)
    deduped: List[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    if not deduped:
        deduped = ["Protagonist", "Companion", "Antagonist"]
    while len(deduped) < 3:
        deduped.append(["Companion", "Antagonist"][len(deduped) - 1])
    return deduped[:3]


def _title_from_prompt(prompt: str) -> str:
    words = re.findall(r"[A-Za-z0-9']+", prompt)
    if not words:
        return "Episode Draft"
    return " ".join(w.capitalize() for w in words[:7])


def _infer_genre(prompt: str) -> str:
    p = prompt.lower()
    if any(k in p for k in ["murder", "detective", "secret", "mystery", "crime"]):
        return "Mystery Drama"
    if any(k in p for k in ["space", "alien", "future", "robot"]):
        return "Sci-Fi Drama"
    if any(k in p for k in ["school", "friendship", "family", "love"]):
        return "Character Drama"
    return "Drama"


def create_episode_plan(prompt: str) -> Dict[str, object]:
    chars = _extract_characters(prompt)
    genre = _infer_genre(prompt)
    title = _title_from_prompt(prompt)

    return {
        "title": title,
        "genre": genre,
        "setting": "Small town, early morning",
        "characters": chars,
        "premise": prompt.strip(),
        "theme": "Trust under pressure",
        "conflict": f"{chars[0]} finds evidence that threatens the group.",
        "turning_point": f"{chars[1]} reveals a hidden motive that changes the investigation.",
        "resolution": f"{chars[0]} and {chars[2]} choose a risky truth over an easy lie.",
    }


def section_prompts(plan: Dict[str, object]) -> Dict[str, str]:
    c1, c2, c3 = plan["characters"]
    premise = plan["premise"]
    conflict = plan["conflict"]
    turning = plan["turning_point"]
    resolution = plan["resolution"]
    theme = plan["theme"]

    return {
        "logline": f"{premise} {c1} senses that nothing is as it seems.",
        "setup": f"At dawn, {c1} tries to keep the day ordinary.",
        "inciting_incident": conflict,
        "rising_action": f"{c1} and {c2} follow signs that point to danger. {turning}",
        "climax": f"{c1} confronts {c3} when the truth can no longer be delayed.",
        "resolution": resolution,
        "scene_1": f"{c1} walks toward the woods and notices unusual tracks.",
        "scene_2": f"{c2} uncovers a detail that changes the meaning of the clues.",
        "scene_3": f"{c1} and {c3} clash over what really happened.",
        "scene_4": f"{c1} makes a final choice and accepts the consequences.",
        "dialogue_a": f"{c1} whispers that the evidence was planted.",
        "dialogue_b": f"{c2} admits they hid one critical fact.",
        "dialogue_c": f"{c3} warns that speaking out will destroy them all.",
    }
