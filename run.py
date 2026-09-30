"""Rank 2+ touchdown candidates for an NFL week.

    python run.py                     # next unplayed week, Jev if TYPESAFE_API_KEY is set
    python run.py --week 4 --no-jev   # numbers-only baseline
    python run.py --week 3 --backtest # where last week's actual 2+ TD scorers ranked
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from venom2td import data, features as F, jev, model, notes

REPORTS = Path(__file__).resolve().parent / "reports"


def next_week(season: int) -> int:
    s = data.schedule()
    s = s[(s.season == season) & (s.game_type == "REG")]
    return int(s[s.result.isna()].week.min())


def fmt_table(d: pd.DataFrame, final: bool) -> str:
    p = "p_2plus_final" if final else "p_2plus"
    lines = ["| # | Player | Pos | Matchup | Kickoff | Inside-5 carries | Inside-10 targets | Exp. TDs | P(2+ TD) | Fair odds |"
             + (" Jev adj. |" if final else ""),
             "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if final else "")]
    for i, r in enumerate(d.itertuples(), 1):
        where = "vs" if r.home else "@"
        flag = f" ({r.report_status})" if isinstance(r.report_status, str) else ""
        lam = r.lam_final if final else r.lam
        prob = getattr(r, p)
        lines.append(
            f"| {i} | {r.player}{flag} | {r.position} | {r.team} {where} {r.opponent} | {r.kickoff} | "
            f"{r.carries_i5}/{r.team_i5_carries} | {r.targets_i10} | {lam:.2f} | {prob:.1%} | "
            f"{F.american_odds(prob)} |" + (f" x{r.jev_mult:.2f} |" if final else "")
        )
    return "\n".join(lines)


def backtest(season: int, week: int, ranked: pd.DataFrame) -> str:
    opp = F.opportunities(data.pbp(season))
    wk = opp[opp.week == week].groupby(["player_id", "player"]).td.sum()
    hits = wk[wk >= 2].reset_index()
    ranked = ranked.reset_index(drop=True)
    out = [f"Actual 2+ TD scorers in week {week} and where the model ranked them beforehand:"]
    for h in hits.itertuples():
        idx = ranked.index[ranked.player_id == h.player_id]
        rank = f"#{idx[0] + 1} ({ranked.p_2plus[idx[0]]:.1%})" if len(idx) else "not in pool"
        out.append(f"- {h.player}: {h.td} TDs, ranked {rank}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int)
    ap.add_argument("--candidates", type=int, default=30, help="how many to send to Jev")
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--no-jev", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="re-download nflverse data")
    a = ap.parse_args()

    week = a.week or next_week(a.season)
    pool = model.baseline(a.season, week, refresh=a.refresh)
    if a.backtest:
        print(backtest(a.season, week, pool))
        return

    cands = pool.head(a.candidates)
    use_jev = not a.no_jev and jev.available()
    if use_jev:
        venom = notes.load(a.season, week)
        cands = jev.judge(cands, pool, venom)
    elif not a.no_jev:
        print("TYPESAFE_API_KEY not set: showing numbers-only baseline.\n")

    title = f"# 2+ TD board: {a.season} week {week}"
    body = fmt_table(cands.head(a.show), use_jev)
    report = f"{title}\n\n{'Jev-adjusted' if use_jev else 'Baseline (no Jev)'}\n\n{body}\n"
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"{a.season}_week{week:02d}{'' if use_jev else '_baseline'}.md"
    out.write_text(report)
    print(report)
    print(f"Saved {out.relative_to(REPORTS.parent)}")


if __name__ == "__main__":
    main()
