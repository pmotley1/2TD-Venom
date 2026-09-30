# 2TD-Venom

Ranks NFL players by their chance to score **2+ touchdowns** in a week.

## How it works

1. **Numbers (code)** – nflverse play-by-play. Every carry and target is valued by the
   league TD rate for its field position (a carry from the 2 is worth far more than one
   from the 40). A player's share of his team's expected TDs, blended with last season's
   role, is multiplied by his team's expected TDs this week (from the betting total and
   spread) and the opponent's TD-allowed rate. P(2+) comes from a Poisson model.
2. **Judgment (TypeSafe Jev)** – for the top candidates, Jev reads the situation in plain
   words and scores goal-line role security, game flow, matchup, injury risk, and your
   Venom notes. Code turns those into a bounded multiplier (x0.7 to x1.3). Weights are in
   `venom2td/jev.py` and are untuned.
3. **Venom (manual)** – paste what Venom's NFL section shows into
   `venom_notes/<season>_week<NN>.md`. Venom's terms forbid automated access, so this is by hand.

## Run

```sh
pip install -r requirements.txt
export TYPESAFE_API_KEY=...        # or set it in the environment settings
python run.py                      # next unplayed week, with Jev if the key is set
python run.py --no-jev             # numbers-only baseline
python run.py --week 3 --backtest  # where that week's actual 2+ TD scorers ranked
python run.py --refresh            # re-download data (injury reports update through the week)
```

Reports are written to `reports/`.

Notes: 2+ TD props count rushing and receiving TDs only (not QB passing TDs). Injury
designations (Questionable/Doubtful/Out) are posted Friday, so rerun late in the week.
