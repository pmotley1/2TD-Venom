"""TypeSafe Jev judgments layered on the numeric baseline.

Jev reads each candidate's situation (described in words, not raw math) and
answers narrow questions. Code turns those answers into a bounded multiplier on
the baseline expected TDs. Weights below are explicit and untuned: adjust them
as results come in.
"""
from __future__ import annotations

import asyncio
import math
import os

import pandas as pd
from typesafe_sdk import AsyncTypeSafeClient, Noul, Score

MODEL = "jev-latest"
CONCURRENCY = 8

# Each Score is 0..4; (score - 2) / 2 maps it to -1..+1 before weighting.
WEIGHTS = {"goal_line_role": 0.15, "game_flow": 0.10, "matchup": 0.08, "venom_support": 0.20}
AVAILABILITY_PENALTY = 0.5   # at availability_risk = 1, expected TDs are halved
MULT_BOUNDS = (0.7, 1.3)

QUESTIONS = {
    "goal_line_role": Score(
        instructions=(
            "How secure is the role of the player in `player` as his team's first option "
            "for touchdowns close to the goal line this week? Use `role_near_goal_line`, "
            "`teammates` and `injury_report`."
        ),
        criteria=[
            "Rarely used near the goal line; teammates get those chances",
            "Occasional option near the goal line, clearly behind a teammate",
            "Splits goal-line chances roughly evenly with teammates",
            "Primary goal-line option, with some sharing",
            "Clear, dominant goal-line option who gets nearly all the chances",
        ],
    ),
    "game_flow": Score(
        instructions=(
            "Given `game_outlook` and the player's position, how favorable is the expected "
            "game flow for this player getting many scoring chances? Running backs benefit when "
            "their team is expected to lead and score a lot; pass catchers benefit from high "
            "scoring games in general."
        ),
        criteria=[
            "Very unfavorable: low-scoring team or game flow that takes the player off the field",
            "Somewhat unfavorable",
            "Neutral",
            "Somewhat favorable",
            "Very favorable: high-scoring team and game flow that feeds this player",
        ],
    ),
    "matchup": Score(
        instructions=(
            "How favorable is `opponent_defense` for the way this player usually scores "
            "(rushing or receiving, as described in `role_near_goal_line`)?"
        ),
        criteria=[
            "Very tough: the defense is strong against exactly how this player scores",
            "Somewhat tough",
            "Average",
            "Somewhat soft",
            "Very soft: the defense is weak against exactly how this player scores",
        ],
    ),
    "availability_risk": Noul(
        instructions=(
            "Based on `injury_report`, is there a real risk that the player misses this game "
            "or plays a clearly reduced role?"
        ),
        criteria={
            "true": "Injury or practice status suggests he may miss the game or be limited",
            "false": "Not on the report, or practiced fully / listed for non-injury reasons",
        },
    ),
}

VENOM_QUESTIONS = {
    "venom_relevant": Noul(
        instructions="Do `venom_notes` contain information about this player's chances of scoring this week?",
    ),
    "venom_support": Score(
        instructions=(
            "How strongly do `venom_notes` support the player in `player` scoring two or more "
            "touchdowns this week?"
        ),
        criteria=[
            "Strongly against: the notes point to a poor scoring outlook",
            "Somewhat against",
            "Neutral or mixed",
            "Somewhat supportive",
            "Strongly supportive: the notes point to an excellent multi-touchdown outlook",
        ],
    ),
}


def _level(value: float, cuts: list[float], words: list[str]) -> str:
    for cut, word in zip(cuts, words):
        if value < cut:
            return word
    return words[-1]


def build_state(row: pd.Series, teammates: pd.DataFrame, venom_note: str | None) -> dict:
    """Describe the player's situation in words for Jev (numbers pre-digested in code)."""
    fav = row.spread
    spread_txt = (f"favored by {fav:g} points" if fav > 0 else
                  f"an underdog by {-fav:g} points" if fav < 0 else "a pick'em")
    total_word = _level(row.total, [41, 45, 49], ["low", "below average", "above average", "high"])
    implied_word = _level(row.implied_pts, [19, 22.5, 26], ["low", "below average", "above average", "high"])
    def_word = ["well below average", "below average", "about average", "above average", "well above average"]
    cuts = [0.8, 0.93, 1.07, 1.2]

    mates = teammates[(teammates.team == row.team) & (teammates.player_id != row.player_id)
                      & (teammates.position == row.position)].nlargest(3, "lam")
    mate_txt = [
        f"{m.player}: {m.carries_i5} carries inside the 5, {m.targets_i10} targets inside the 10"
        + (f", injury status: {m.report_status}" if isinstance(m.report_status, str) else "")
        for m in mates.itertuples()
    ] or ["No other notable players at his position"]

    practice = row.practice_status if isinstance(row.practice_status, str) else None
    report = row.report_status if isinstance(row.report_status, str) else None
    injury = row.practice_primary_injury if isinstance(row.practice_primary_injury, str) else None
    if practice or report:
        inj_txt = f"Game status: {report or 'not yet designated'}. Practice: {practice or 'n/a'}. Injury: {injury or 'n/a'}."
    else:
        inj_txt = "Not on this week's injury report."

    return {
        "player": {"name": row.player, "position": row.position, "team": row.team,
                   "opponent": row.opponent, "home_game": bool(row.home)},
        "role_near_goal_line": (
            f"Over {row.games} games this season: {row.carries_i5} of his team's "
            f"{row.team_i5_carries} carries inside the 5-yard line, {row.carries_i10} carries inside "
            f"the 10, {row.targets_i10} targets inside the 10, {row.targets_rz} red-zone targets. "
            f"Scored {row.rush_td} rushing and {row.rec_td} receiving touchdowns; "
            f"{row.multi_td_games} games with 2+ touchdowns."
        ),
        "teammates": mate_txt,
        "game_outlook": (
            f"His team is {spread_txt}. The game total is {total_word}, and his team's implied "
            f"point total is {implied_word}. Roof: {row.roof if isinstance(row.roof, str) else 'unknown'}."
        ),
        "opponent_defense": (
            f"This season {row.opponent} has allowed rushing touchdowns at a rate that is "
            f"{_level(row.def_rush, cuts, def_word)}, and receiving touchdowns at a rate that is "
            f"{_level(row.def_rec, cuts, def_word)}."
        ),
        "injury_report": inj_txt,
        **({"venom_notes": venom_note} if venom_note else {}),
    }


def multiplier(answers: dict) -> float:
    z = sum(WEIGHTS[k] * (answers[k] - 2) / 2 for k in ("goal_line_role", "game_flow", "matchup"))
    if pd.notna(answers.get("venom_support")):
        z += WEIGHTS["venom_support"] * answers["venom_relevant"] * (answers["venom_support"] - 2) / 2
    m = math.exp(z) * (1 - AVAILABILITY_PENALTY * max(0.0, answers["availability_risk"] - 0.5) * 2)
    return min(max(m, MULT_BOUNDS[0]), MULT_BOUNDS[1])


async def _judge_all(states: list[dict]) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY)
    async with AsyncTypeSafeClient(model=MODEL) as client:
        async def one(state: dict) -> dict:
            qs = {**QUESTIONS, **(VENOM_QUESTIONS if "venom_notes" in state else {})}
            async with sem:
                res = await client.system_one(state, qs)
            out = {k: a.score for k, a in res.scores.items()}
            out.update({f"{k}_conf": a.confidence for k, a in res.scores.items()})
            out.update({k: a.noul for k, a in res.nouls.items()})
            out["jev_model"] = res.model
            return out
        return await asyncio.gather(*(one(s) for s in states))


def available() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY"))


def judge(cands: pd.DataFrame, pool: pd.DataFrame, notes: dict[str, str]) -> pd.DataFrame:
    """Ask Jev about each candidate and apply the multiplier to expected TDs."""
    from .features import prob_2plus
    from .notes import note_for

    states = [build_state(r, pool, note_for(notes, r.player)) for _, r in cands.iterrows()]
    answers = pd.DataFrame(asyncio.run(_judge_all(states)), index=cands.index)
    out = cands.join(answers)
    out["jev_mult"] = [multiplier(a) for a in answers.to_dict("records")]
    out["lam_final"] = out.lam * out.jev_mult
    out["p_2plus_final"] = prob_2plus(out.lam_final)
    out["jev_state"] = states
    return out.sort_values("p_2plus_final", ascending=False)
