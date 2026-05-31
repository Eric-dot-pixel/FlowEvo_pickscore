# PickScore Results Comparison: r0004 vs ODE / Best-of-4

## Scope

- Compared methods: `r0004` best workflow controller, `ode` baseline, `bestof4` baseline.
- Splits: `train` and `test`.
- `r0004 train` is taken from the archived round-4 workflow evaluation summary.
- `r0004 test` is taken from the dedicated full-test evaluation summary.
- Baseline summaries come from the compact aggregate summaries under `train_*_baseline` and `test_*_baseline`.

## Train

- Sample size: `500`

| beta | r0004 nfe | r0004 reward | ode nfe | ode reward | r0004 - ode | bestof4 nfe | bestof4 reward | r0004 - bestof4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 10.000 | 0.787971 | 8.000 | 0.809376 | -0.021405 | 8.000 | 0.676462 | +0.111510 |
| 0.25 | 20.000 | 0.817857 | 20.000 | 0.828298 | -0.010441 | 20.000 | 0.802083 | +0.015774 |
| 0.50 | 36.000 | 0.837384 | 36.000 | 0.829421 | +0.007963 | 36.000 | 0.836650 | +0.000735 |
| 0.75 | 47.430 | 0.840149 | 48.000 | 0.829238 | +0.010911 | 48.000 | 0.843990 | -0.003841 |
| 1.00 | 63.554 | 0.843016 | 64.000 | 0.829445 | +0.013571 | 64.000 | 0.847290 | -0.004274 |

| summary | beats ode | beats bestof4 | total delta vs ode | min delta vs ode | total delta vs bestof4 | min delta vs bestof4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| value | 3 | 3 | +0.000599 | -0.021405 | +0.119904 | -0.004274 |

## Test

- Sample size: `2048`

| beta | r0004 nfe | r0004 reward | ode nfe | ode reward | r0004 - ode | bestof4 nfe | bestof4 reward | r0004 - bestof4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 10.000 | 0.793020 | 8.000 | 0.812158 | -0.019138 | 8.000 | 0.678004 | +0.115017 |
| 0.25 | 20.000 | 0.821337 | 20.000 | 0.833330 | -0.011993 | 20.000 | 0.804293 | +0.017044 |
| 0.50 | 36.000 | 0.840123 | 36.000 | 0.834273 | +0.005850 | 36.000 | 0.839663 | +0.000460 |
| 0.75 | 47.434 | 0.843393 | 48.000 | 0.834362 | +0.009031 | 48.000 | 0.847570 | -0.004177 |
| 1.00 | 63.561 | 0.845914 | 64.000 | 0.833701 | +0.012213 | 64.000 | 0.851227 | -0.005314 |

| summary | beats ode | beats bestof4 | total delta vs ode | min delta vs ode | total delta vs bestof4 | min delta vs bestof4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| value | 3 | 3 | -0.004037 | -0.019138 | +0.123030 | -0.005314 |

## Notes

- `r0004` beats `ode` at `beta=0.5/0.75/1.0`, but loses at `beta=0.0/0.25`, on both train and test.
- Against `bestof4`, `r0004` also wins 3 out of 5 betas on both splits: it is clearly better at `beta=0.0/0.25` and slightly better at `beta=0.5`, but loses at `beta=0.75/1.0`.
- On train, `r0004` is almost tied with `ode` in total reward gap across the 5 betas: `+0.000599`.
- On test, `r0004` is slightly behind `ode` in total reward gap across the 5 betas: `-0.004037`, despite winning 3 out of 5 betas, because the low-beta losses are larger than the mid/high-beta gains.
- Relative to both baselines, `r0004` uses more compute at `beta=0.0` because the workflow target for the iterative experiment was fixed at `10` NFE, while both compact baselines at `beta=0.0` used `8` NFE.
