# RIVERLOAD C2 — forecast the Rhine, then load the ship

**2nd place, Odyssia Datathon, September 2026.** Two coupled problems in one:
forecast how deep the Rhine will be over the next five days, then decide what
to put on a ship, which route to sail and when to leave — under constraints
that the forecast itself determines.

My code only. The organisers' dataset and Python package are **not**
redistributed here.

---

## Why the two halves cannot be solved separately

The river's depth at each of 23 segments is `guaranteed depth + (gauge level −
reference level)`. The shallowest point on a route sets the maximum draught,
the draught sets how much cargo the ship can carry, and the cargo determines
revenue minus late-delivery penalties.

So the forecast is not the deliverable — it is a **constraint generator** for
the planner. And that changes what a good forecast looks like.

**A point forecast is the wrong output.** If you predict the minimum depth
`dmin` and you are 10 cm too high, the plan you build on it loads the ship too
heavily and it grounds. Too low, and you leave paid cargo on the quay. The two
errors cost wildly different amounts, so the loss is **asymmetric** and a model
trained on squared error optimises the wrong thing.

Everything below follows from that.

## Forecasting

**Quantile regression, not regression.** LightGBM with
`objective='quantile'`, evaluated with **pinball loss**, producing a
distribution of plausible depths rather than one number. The planner can then
ask for a conservative quantile where grounding risk dominates and a central
one where it does not.

**Predict the change, not the level.** The target is `y = dmin − now`: how much
the minimum depth will move from where the river is at issue time. The absolute
level is dominated by which gauge and season you are in, which the model would
otherwise spend all its capacity learning. The delta is the part that is
actually uncertain.

**Three forecasters, blended.** `analog_predict` finds historical analogues —
past river states resembling the current one — and takes a weighted quantile of
what happened next. `state_predict` conditions on the current hydrological
state. `boosted_predict` is the gradient-boosted model on engineered features.
They fail differently: analogues are strong when the present looks like the
past and useless in an unprecedented state, while the boosted model
extrapolates but can drift. `blend_forecasts` combines them.

**Strictly chronological backtesting.** Validation is leave-one-year-out and
forward-only, never a random split. A random split over a time series lets the
model see 2021 while predicting 2019, and the river is autocorrelated over
weeks — so a shuffled split would report a score that could never be achieved
in production. `fit_calibration` then corrects the quantiles on held-out
predictions, grouped, because a quantile model that is not calibrated is not
giving you the probability it claims.

Features come only from the supplied history — hourly levels at 21 gauges back
to 2000 and 16 discharge series. **No external hydrological data**, which the
header of `riverload_c2.py` states, since it was a competition condition.

## Planning

With a depth distribution in hand, the decision is a search over **2 routes ×
25 departure slots**, and for each of those, which containers from which ports
to load into which of the ship's 48 stacks.

**Expected cost, not best case.** Each route/slot pair is scored against the
forecast *distribution* — a stochastic expected cost — rather than against a
single predicted river. A departure that is excellent if the river holds up and
catastrophic if it drops is correctly ranked below a duller one that is fine
either way.

**Mixed-integer programming for the cargo.** Container selection across
multiple ports is a knapsack with side constraints: container masses, values,
per-hour late penalties, unloading order (what is loaded first comes off last),
and the ship's **trim** — its 48 stacks have different lever arms, so where
mass sits matters, not just how much. That goes to `scipy.optimize.milp` with
explicit `LinearConstraint`s and a **0.6 s time budget per instance**, because
94 test scenarios × 2 routes × 25 slots means the solver is called thousands of
times and an exact answer that takes a minute is useless. Six finalist
route/slot candidates survive to full stowage validation.

**Tiered stowage, validated.** A plan is only accepted if it passes the
organisers' own feasibility rules, so the submission never contains a loading
that could not physically sail.

## Layout

```
riverload_c2.py                            the complete solution, 2,520 lines
notebooks/
  00-kaggle-one-cell-submission.ipynb      the single-cell offline submission
  01-first-model-quantile-lgbm.ipynb       quantile LightGBM baseline
  02-quantile-lgbm-revised.ipynb
  03-trajectory-ensemble.ipynb             trajectory ensembles + analogues
  04-trajectory-ensemble-revised.ipynb
```

It is written to run offline in one cell because the competition's execution
environment had no internet access. Determinism is pinned:
`PYTHONHASHSEED=2026`, fixed thread counts, `MILP_TIME_SECONDS=0.60`.

## Running it

You need the organisers' `riverload-data` package (reference geometry, 23 river
segments, 21 gauges, hourly history from 2000, 596 training scenarios and 94
test scenarios) plus their `riverload` Python package, which contains the
timetable, depth, ship and exact scoring code. Neither is redistributed here —
they came from the Odyssia organisers and are theirs.

`riverload_c2.py` expects the data zip and unpacks it; the notebooks mount it
from Drive, which is how they ran during the event.

```console
python riverload_c2.py          # expects riverload-data alongside
```

## Known limits

- **The forecast ensemble is not ablated.** I blend three predictors and
  measured the blend, not each one's standalone contribution. I cannot tell you
  which of the three earned the placing.
- **The MILP time budget is tuned by hand.** 0.6 s was chosen to fit the
  scenario count inside the runtime limit, not derived from an
  optimality-gap target.
- **No uncertainty on the planner side.** The forecast is probabilistic; the
  stowage decision, once a route and slot are chosen, is treated as
  deterministic.
- **One-cell structure is a competition artifact, not a design.** 2,520 lines
  in one file is how you ship into a notebook with no internet and a runtime
  cap. It is not how this should be organised, and splitting it into modules is
  the first thing I would do before reusing any of it.
- **Cross-validation is leave-one-year-out**, so with seven years of scenarios
  the variance on any single estimate is not small.

## Provenance

The RIVERLOAD data, the river geometry and cost constants, the organisers'
`riverload` package and the official scorer belong to the **Odyssia Datathon
organisers**. Only my own forecasting and planning code is published here.

The datathon's other track — breast tumour detection and segmentation from
ultrasound, with risk and survival estimation from clinical and gene-expression
data — is deliberately **not** in this repository.

MIT, for my code only.
