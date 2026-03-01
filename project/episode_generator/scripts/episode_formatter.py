from __future__ import annotations

import re
from typing import Dict


def _normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text[0].upper() + text[1:]


def _safe(text: str, fallback: str) -> str:
    t = _normalize(text)
    return t if t else fallback


def compose_episode_from_plan(
    plan: Dict[str, object],
    sections: Dict[str, str],
    model_name: str,
) -> str:
    chars = plan["characters"]

    logline = _safe(
        sections.get("logline", ""),
        f"{chars[0]} faces a growing mystery that tests loyalty and courage.",
    )

    lines = [
        f"TITLE: {plan['title']}",
        f"MODEL: {model_name}",
        "",
        "EPISODE ITEMS",
        f"- Genre: {plan['genre']}",
        f"- Setting: {plan['setting']}",
        f"- Main Characters: {chars[0]}, {chars[1]}, {chars[2]}",
        f"- Theme: {plan['theme']}",
        "",
        "LOGLINE",
        logline,
        "",
        "PLOT OUTLINE",
        f"1. Setup: {_safe(sections.get('setup', ''), str(plan['premise']))}",
        f"2. Inciting Incident: {_safe(sections.get('inciting_incident', ''), str(plan['conflict']))}",
        f"3. Rising Action: {_safe(sections.get('rising_action', ''), str(plan['turning_point']))}",
        f"4. Climax: {_safe(sections.get('climax', ''), f'{chars[0]} is forced to choose between safety and truth.')}",
        f"5. Resolution: {_safe(sections.get('resolution', ''), str(plan['resolution']))}",
        "",
        "SCENE BREAKDOWN",
        f"Scene 1 - Opening: {_safe(sections.get('scene_1', ''), 'The day begins with an uneasy calm.')}",
        f"Scene 2 - Discovery: {_safe(sections.get('scene_2', ''), 'A clue reveals hidden tension.')}",
        f"Scene 3 - Confrontation: {_safe(sections.get('scene_3', ''), 'Allies clash over what to do next.')}",
        f"Scene 4 - Closing Beat: {_safe(sections.get('scene_4', ''), 'A fragile victory hints at future fallout.')}",
        "",
        "SCRIPT SAMPLE",
        "INT. EDGE OF TOWN - MORNING",
        f"{chars[0]} studies tracks near the tree line while the wind carries distant voices.",
        f"{chars[0]}: \"{_safe(sections.get('dialogue_a', ''), 'Something happened here before sunrise.')}\"",
        f"{chars[1]}: \"{_safe(sections.get('dialogue_b', ''), 'If we follow this clue, everything changes.')}\"",
        f"{chars[2]}: \"{_safe(sections.get('dialogue_c', ''), 'Then decide now, because we are out of time.')}\"",
    ]

    return "\n".join(lines)
