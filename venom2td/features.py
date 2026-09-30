"""Turn play-by-play into scoring-opportunity features.

Everything numeric lives here (Jev is weak at arithmetic, so it never sees raw
math problems). The core idea: a touchdown is mostly a function of *where* a
player gets the ball. A carry from the 2 scores far more often than a carry
from the 40, so we value every carry and target by the league touchdown rate
for its field-position bucket ("expected TDs", xTD).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# yardline_100 = yards from the opponent's end zone
BUCKETS = [0, 2, 5, 10, 20, 100]
BUCKET_LABELS = ["1-2", "3-5", "6-10", "11-20", "21+"]


def opportunities(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per carry or target, with who got it and whether it scored."""
    p = pbp[(pbp.season_type == "REG") & (pbp.two_point_attempt != 1)]
    p = p[p.yardline_100.notna() & p.posteam.notna()]

    rush = p[(p.rush_attempt == 1) & p.rusher_player_id.notna() & (p.qb_kneel != 1)]
    rush = rush.assign(
        kind="rush",
        player_id=rush.rusher_player_id,
        player=rush.rusher_player_name,
        td=(rush.rush_touchdown == 1) & (rush.td_player_id == rush.rusher_player_id),
    )
    tgt = p[(p.pass_attempt == 1) & p.receiver_player_id.notna() & (p.sack != 1)]
    tgt = tgt.assign(
        kind="rec",
        player_id=tgt.receiver_player_id,
        player=tgt.receiver_player_name,
        td=(tgt.pass_touchdown == 1) & (tgt.td_player_id == tgt.receiver_player_id),
    )
    opp = pd.concat([rush, tgt])[
        ["season", "week", "game_id", "posteam", "defteam", "player_id", "player",
         "kind", "yardline_100", "td"]
    ]
    opp["bucket"] = pd.cut(opp.yardline_100, BUCKETS, labels=BUCKET_LABELS)
    opp["td"] = opp.td.astype(int)
    return opp


def td_rates(opp: pd.DataFrame) -> pd.Series:
    """League TD rate per (kind, bucket); the xTD value of one opportunity."""
    return opp.groupby(["kind", "bucket"], observed=True).td.mean()


def with_xtd(opp: pd.DataFrame, rates: pd.Series) -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays([opp.kind, opp.bucket])
    return opp.assign(xtd=rates.reindex(idx).to_numpy())


def tds_per_point(pbp: pd.DataFrame, sched: pd.DataFrame, season: int) -> float:
    """Offensive rushing+receiving TDs per point scored (converts implied totals to TDs)."""
    opp = opportunities(pbp)
    g = sched[(sched.season == season) & (sched.game_type == "REG") & sched.result.notna()]
    points = g.home_score.sum() + g.away_score.sum()
    return opp.td.sum() / points


def player_table(opp: pd.DataFrame) -> pd.DataFrame:
    """Per player-team totals for the window, plus share of team xTD in games played."""
    games = opp.groupby(["posteam", "player_id"]).game_id.unique()
    team_game = opp.groupby(["posteam", "game_id"]).agg(team_xtd=("xtd", "sum"), team_td=("td", "sum"))

    def agg(d: pd.DataFrame) -> pd.Series:
        r, t = d[d.kind == "rush"], d[d.kind == "rec"]
        return pd.Series({
            "player": d.player.mode().iat[0],
            "games": d.game_id.nunique(),
            "carries": len(r),
            "carries_i5": int((r.yardline_100 <= 5).sum()),
            "carries_i10": int((r.yardline_100 <= 10).sum()),
            "targets": len(t),
            "targets_i10": int((t.yardline_100 <= 10).sum()),
            "targets_rz": int((t.yardline_100 <= 20).sum()),
            "rush_td": int(r.td.sum()),
            "rec_td": int(t.td.sum()),
            "xtd_rush": r.xtd.sum(),
            "xtd_rec": t.xtd.sum(),
            "multi_td_games": int((d.groupby("game_id").td.sum() >= 2).sum()),
        })

    pt = opp.groupby(["posteam", "player_id"]).apply(agg, include_groups=False)
    tg = team_game.reset_index()
    team_in_games = [
        tg[(tg.posteam == team) & tg.game_id.isin(games[(team, pid)])][["team_xtd", "team_td"]].sum()
        for team, pid in pt.index
    ]
    pt[["team_xtd", "team_td"]] = pd.DataFrame(team_in_games, index=pt.index)
    pt["team_i5_carries"] = [
        int(((opp.posteam == team) & (opp.kind == "rush") & (opp.yardline_100 <= 5)
             & opp.game_id.isin(games[(team, pid)])).sum())
        for team, pid in pt.index
    ]
    return pt.reset_index().rename(columns={"posteam": "team"})


def defense_factors(opp: pd.DataFrame, shrink_games: float = 5.0) -> pd.DataFrame:
    """How many rush/rec TDs each defense allows vs league average (1.0 = average).

    Heavily shrunk toward 1.0 early in the season: 3 games of defense is noise.
    """
    per = opp.groupby(["defteam", "kind"]).td.sum().unstack(fill_value=0)
    gp = opp.groupby("defteam").game_id.nunique()
    league = per.sum() / gp.sum()
    out = pd.DataFrame(index=per.index)
    for kind in ["rush", "rec"]:
        out[f"def_{kind}"] = (per[kind] + shrink_games * league[kind]) / (gp + shrink_games) / league[kind]
    out["def_games"] = gp
    return out


def implied_totals(sched: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Team implied points from the game total and spread (nflverse: spread > 0 = home favored)."""
    g = sched[(sched.season == season) & (sched.week == week) & (sched.game_type == "REG")]
    rows = []
    for _, r in g.iterrows():
        home = (r.total_line + r.spread_line) / 2
        for team, opp, pts, fav, is_home in [
            (r.home_team, r.away_team, home, r.spread_line, True),
            (r.away_team, r.home_team, r.total_line - home, -r.spread_line, False),
        ]:
            rows.append({
                "team": team, "opponent": opp, "home": is_home, "game_id": r.game_id,
                "kickoff": f"{r.weekday} {r.gameday} {r.gametime}", "total": r.total_line,
                "spread": fav, "implied_pts": pts, "roof": r.roof,
            })
    return pd.DataFrame(rows)


def injury_status(inj: pd.DataFrame, week: int) -> pd.DataFrame:
    w = inj[inj.week == week]
    cols = ["gsis_id", "report_status", "practice_status", "report_primary_injury", "practice_primary_injury"]
    return w[cols].drop_duplicates("gsis_id", keep="last").rename(columns={"gsis_id": "player_id"})


def prob_2plus(lam: np.ndarray | float) -> np.ndarray | float:
    """P(2+ TDs) under a Poisson with mean lam."""
    return 1 - np.exp(-lam) * (1 + lam)


def american_odds(p: float) -> str:
    if p <= 0 or p >= 1:
        return "n/a"
    return f"-{round(100 * p / (1 - p))}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"
