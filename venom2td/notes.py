"""Load Venom notes you copy in by hand (Venom's terms forbid automated access).

One file per week: venom_notes/2026_week04.md. Put each player under a
level-2 heading with his full name, and paste whatever Venom shows for him:

    ## Jahmyr Gibbs
    Venom score 91, red zone share ..., matchup grade ...

A "## General" section, if present, is added to every player's notes.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

NOTES_DIR = Path(__file__).resolve().parent.parent / "venom_notes"


def _key(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name.lower())
    return re.sub(r"[^a-z]", "", name)


def path_for(season: int, week: int) -> Path:
    return NOTES_DIR / f"{season}_week{week:02d}.md"


def load(season: int, week: int) -> dict[str, str]:
    path = path_for(season, week)
    if not path.exists():
        return {}
    notes: dict[str, str] = {}
    for block in re.split(r"^## +", path.read_text(), flags=re.M)[1:]:
        title, _, body = block.partition("\n")
        if body.strip():
            notes[_key(title)] = body.strip()
    return notes


def note_for(notes: dict[str, str], player: str) -> str | None:
    parts = [notes.get(_key(player)), notes.get("general")]
    text = "\n\n".join(p for p in parts if p)
    return text or None
