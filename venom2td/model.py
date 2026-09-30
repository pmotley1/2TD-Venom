"""Baseline 2+ TD probabilities from opportunity share, team implied total and defense."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, features as F

PRIOR_GAMES = 3.0      # how many games of weight last season's role gets
NO_PRIOR_SHARE = 0.08  # generic share assumed for players with no prior-season role
XTD_WEIGHT = 0.75      # opportunity-based share vs actual-TD share (TDs are noisy)
TD_SHARE_SHRINK = 6.0  # team TDs needed before actual-TD share gets half its weight
SKILL_POS = {"RB", "WR", "TE", "QB", "FB"}


def _shares(pt: pd.DataFrame) -> pd.DataFrame:
    team_xtd = pt.team_xtd.replace(0, np.nan)
    team_td = pt.team_td.replace(0, np.nan)
    x_rush, x_rec = pt.xtd_rush / team_xtd, pt.xtd_rec / team_xtd
    a_rush, a_rec = (pt.rush_td / team_td).fillna(x_rush), (pt.rec_td / team_td).fillna(x_rec)
    # Actual-TD share means little when the team has scored only a handful of TDs.
    w_actual = (1 - XTD_WEIGHT) * pt.team_td / (pt.team_td + TD_SHARE_SHRINK)
    return pt.assign(
        rush_share=(1 - w_actual) * x_rush.fillna(0) + w_actual * a_rush.fillna(0),
        rec_share=(1 - w_actual) * x_rec.fillna(0) + w_actual * a_rec.fillna(0),
    )


def baseline(season: int, week: int, refresh: bool = False) -> pd.DataFrame:
    sched = data.schedule(refresh)
    prev_pbp = data.pbp(season - 1)
    cur_pbp = data.pbp(season, refresh)
    cur_pbp = cur_pbp[cur_pbp.week < week]

    prev_opp = F.opportunities(prev_pbp)
    rates = F.td_rates(pd.concat([prev_opp, F.opportunities(cur_pbp)]))
    prev_opp = F.with_xtd(prev_opp, rates)
    cur_opp = F.with_xtd(F.opportunities(cur_pbp), rates)
    tds_per_pt = F.tds_per_point(prev_pbp, sched, season - 1)

    cur = _shares(F.player_table(cur_opp))
    prev = _shares(F.player_table(prev_opp)).sort_values("games").drop_duplicates("player_id", keep="last")
    prev = prev.set_index("player_id")[["rush_share", "rec_share", "games", "rush_td", "rec_td"]].add_prefix("prev_")
    cur = cur.join(prev, on="player_id")

    # Shrink this season's (small-sample) share toward last season's role, or
    # toward a modest generic share for players without a real track record.
    has_prior = cur.prev_games.fillna(0) >= 6
    for c in ["rush_share", "rec_share"]:
        prior = np.where(has_prior, cur[f"prev_{c}"].fillna(0), NO_PRIOR_SHARE * (cur[c] > 0))
        cur[c] = (cur.games * cur[c] + PRIOR_GAMES * prior) / (cur.games + PRIOR_GAMES)

    # Current team and position from the latest weekly roster (handles trades/signings).
    ros = data.rosters(season, refresh)
    ros = ros[ros.week <= week].sort_values("week").drop_duplicates("gsis_id", keep="last")
    ros = ros.set_index("gsis_id")[["team", "position", "status", "full_name"]].rename(columns={"team": "cur_team"})
    cur = cur.join(ros, on="player_id")
    cur["team"] = cur.cur_team.fillna(cur.team)
    cur["player"] = cur.full_name.fillna(cur.player)
    cur = cur[cur.position.isin(SKILL_POS) & (cur.status.fillna("ACT") == "ACT")]
    # A player can appear under two teams after a trade; keep the current-team row.
    cur = cur.sort_values("games").drop_duplicates("player_id", keep="last")

    games = F.implied_totals(sched, season, week)
    cur = cur.merge(games, on="team", how="inner")
    defense = F.defense_factors(cur_opp)
    cur = cur.join(defense[["def_rush", "def_rec"]], on="opponent")
    cur[["def_rush", "def_rec"]] = cur[["def_rush", "def_rec"]].fillna(1.0)

    cur["team_exp_tds"] = cur.implied_pts * tds_per_pt
    cur["lam_rush"] = cur.team_exp_tds * cur.rush_share * cur.def_rush
    cur["lam_rec"] = cur.team_exp_tds * cur.rec_share * cur.def_rec
    cur["lam"] = cur.lam_rush + cur.lam_rec

    inj = F.injury_status(data.injuries(season, refresh), week)
    cur = cur.merge(inj, on="player_id", how="left")
    status = cur.report_status.fillna("")
    cur = cur[~status.isin(["Out", "Injured Reserve"])]
    cur.loc[status == "Doubtful", "lam"] *= 0.25

    cur["p_2plus"] = F.prob_2plus(cur.lam)
    cur["p_anytime"] = 1 - np.exp(-cur.lam)
    return cur.sort_values("p_2plus", ascending=False).reset_index(drop=True)
