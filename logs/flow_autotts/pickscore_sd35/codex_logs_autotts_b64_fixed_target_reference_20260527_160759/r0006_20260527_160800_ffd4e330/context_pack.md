# Flow AutoTTS Context Pack

Read this file first. It is the intended context budget for this round.

## Allowed First-Pass Reads

- `flow_tts_controller_implementation_spec.md`
- `flow_autotts/controllers/optimal.py`
- `flow_autotts/controllers/baselines.py`
- `flow_autotts/core/state.py`
- `flow_autotts/core/errors.py`
- `flow_autotts/experiments/pickscore_sd35/harness.py`
- `flow_autotts/experiments/pickscore_sd35/env.py`
- recent round summaries listed below

## Write Boundary

- Edit only `flow_autotts/controllers/optimal.py`.
- Do not edit the harness, environment, dataset loader, workflow, tests, logs, model directories, or datasets.
- Keep the controller self-contained. The workflow resets it from the template before every round.

## Context Discipline

- Do not run broad repository scans such as `find .` or unconstrained `rg` from repo root.
- Do not bulk-read raw `history.json`, raw event logs, datasets, `SD_3.5_med/`, `PickScore_v1/`, `flow_grpo/`, `.git/`, or `logs/`.
- If a compact summary points to a concrete anomaly, inspect only the relevant small snippet from that round.
- Prefer targeted reads of the files listed above.

## Template

- `flow_autotts/controllers/optimal.template.py`

## Baseline References

These compact baseline files are injected by the workflow so the proposer can compare by nearest NFE.

### `logs/flow_autotts/pickscore_sd35/train_bestof4_ode_retry2_clean_b64_compact_baseline/aggregate_summary.json`

```json
[
  {
    "action_statistics": {
      "answer": 4.0,
      "forward": 8.0,
      "mean_nfe": 8.0,
      "spawn": 4.0
    },
    "behavior_summary": "best-of-4 deterministic ODE (spawn=4.00, forward=8.00, nfe=8.00, single_ode_nfe=2)",
    "beta": 0.0,
    "nfe": 8.0,
    "reward": 0.6764616433382035,
    "reward_per_nfe": 0.08455770541727543
  },
  {
    "action_statistics": {
      "answer": 4.0,
      "forward": 20.0,
      "mean_nfe": 20.0,
      "spawn": 4.0
    },
    "behavior_summary": "best-of-4 deterministic ODE (spawn=4.00, forward=20.00, nfe=20.00, single_ode_nfe=5)",
    "beta": 0.25,
    "nfe": 20.0,
    "reward": 0.8020827637910843,
    "reward_per_nfe": 0.04010413818955422
  },
  {
    "action_statistics": {
      "answer": 4.0,
      "forward": 36.0,
      "mean_nfe": 36.0,
      "spawn": 4.0
    },
    "behavior_summary": "best-of-4 deterministic ODE (spawn=4.00, forward=36.00, nfe=36.00, single_ode_nfe=9)",
    "beta": 0.5,
    "nfe": 36.0,
    "reward": 0.8366495504379272,
    "reward_per_nfe": 0.023240265289942424
  },
  {
    "action_statistics": {
      "answer": 4.0,
      "forward": 48.0,
      "mean_nfe": 48.0,
      "spawn": 4.0
    },
    "behavior_summary": "best-of-4 deterministic ODE (spawn=4.00, forward=48.00, nfe=48.00, single_ode_nfe=12)",
    "beta": 0.75,
    "nfe": 48.0,
    "reward": 0.8439898520708085,
    "reward_per_nfe": 0.01758312191814184
  },
  {
    "action_statistics": {
      "answer": 4.0,
      "forward": 64.0,
      "mean_nfe": 64.0,
      "spawn": 4.0
    },
    "behavior_summary": "best-of-4 deterministic ODE (spawn=4.00, forward=64.00, nfe=64.00, single_ode_nfe=16)",
    "beta": 1.0,
    "nfe": 64.0,
    "reward": 0.8472898569107056,
    "reward_per_nfe": 0.013238904014229775
  }
]
```

## Beta Target Curve

Use the first injected baseline as the beta-matched reward reference for this run.
The target NFE schedule is fixed for this experiment rather than inferred from whatever baseline row happens to be loaded.
For each beta, treat the listed target NFE as a strong alignment reference rather than the optimization target itself.
The real goal is still to push reward above the beta-matched baseline; target NFE is there to keep compute comparable.
Only beta=1.0 has a hard compute limit here: NFE must never exceed 64.

| beta | target_nfe | target_reward | baseline_behavior |
| ---: | ---: | ---: | --- | --- |
| 0.000 | 10.000 | 0.676462 | best-of-4 deterministic ODE (spawn=4.00, forward=8.00, nfe=8.00, single_ode_nfe=2) |
| 0.250 | 20.000 | 0.802083 | best-of-4 deterministic ODE (spawn=4.00, forward=20.00, nfe=20.00, single_ode_nfe=5) |
| 0.500 | 36.000 | 0.836650 | best-of-4 deterministic ODE (spawn=4.00, forward=36.00, nfe=36.00, single_ode_nfe=9) |
| 0.750 | 48.000 | 0.843990 | best-of-4 deterministic ODE (spawn=4.00, forward=48.00, nfe=48.00, single_ode_nfe=12) |
| 1.000 | 64.000 | 0.847290 | best-of-4 deterministic ODE (spawn=4.00, forward=64.00, nfe=64.00, single_ode_nfe=16) |

## Action Semantics And Likely Effects

| action | immediate NFE cost | typical use | what it changes | failure mode |
| --- | ---: | --- | --- | --- |
| `spawn(n)` | 0 | create width cheaply | more active particles at `t=0` | spawning too many weak branches that cannot be advanced or previewed |
| `forward(pid, target_time, solver)` | number of step advances | spend budget to move a branch toward cleaner states | raises time, often improves preview reliability, consumes most of the budget | blindly finishing weak branches without preview evidence |
| `preview(pid)` | 1 | buy a score/uncertainty/drift observation | creates an anchor and evidence for ranking or refinement, but does not advance time | previewing too early or too often without acting on the signal |
| `backward(anchor_id, ...)` | 0 immediate | local refinement or diversity around a promising anchor | creates new children that later need forward/preview budget | branching from weak anchors or creating children that cannot be evaluated |
| `prune(ids)` | 0 | save future budget by removing losers | permanently drops active particles | pruning too aggressively and collapsing diversity |
| `answer(rule='best_preview_score')` | 0 | terminate using best observed anchor | ends the episode without extra rollout cost | answering before enough evidence exists |
| `answer(rule='latest_active')` | auto-forward cost if needed | force-complete the deepest active branch | may spend leftover NFE to reach `t=1` | accidental budget overshoot via implicit final forward steps |

Controller design implication:
- `forward(..., solver=...)` can legally use either `euler` or `sde`; both are available controller choices.
- `forward` and `preview` are the only actions that directly spend NFE in the common path.
- `preview` is the only way to observe score/uncertainty/drift; without it, pruning and backward are evidence-poor.
- `backward` is only useful if the selected anchor is already promising enough to justify spending later NFE on its children.
- If a beta target is being underspent, the safest extra compute is usually selective late `preview`, one more `forward`, or a small local `backward` refinement that can still be evaluated before answering.

## Historical Best Near Beta Target

| beta | target_nfe | best_round | best_nfe | best_reward | delta_vs_beta_target |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0.000 | 10.000 | r0004 | 10.000 | 0.787971 | 0.111510 |
| 0.250 | 20.000 | r0004 | 20.000 | 0.817857 | 0.015774 |
| 0.500 | 36.000 | r0000 | 36.000 | 0.839043 | 0.002394 |
| 0.750 | 48.000 | r0000 | 48.000 | 0.841569 | -0.002421 |
| 1.000 | 64.000 | r0000 | 64.000 | 0.844569 | -0.002721 |

## Recent Round Frontier Comparison

| round | beta | mean_nfe | target_nfe | nfe_gap | nfe_status | reward | beta_target_reward | delta_to_beta_target | nearest_baseline_nfe | delta_to_nearest | actions |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| r0004 | 0.000 | 10.000 | 10.000 | 0.000 | on target | 0.787971 | 0.676462 | 0.111510 | 8.000 | 0.111510 | single-root preview (spawn=1.00, preview=3.00, prune=1.00, nfe=10.00) |
| r0004 | 0.250 | 20.000 | 20.000 | 0.000 | on target | 0.817857 | 0.802083 | 0.015774 | 20.000 | 0.015774 | single-root preview (spawn=1.00, preview=7.00, prune=1.15, nfe=20.00) |
| r0004 | 0.500 | 36.000 | 36.000 | 0.000 | on target | 0.837384 | 0.836650 | 0.000735 | 36.000 | 0.000735 | single-root preview (spawn=1.00, preview=9.00, prune=1.18, nfe=36.00) |
| r0004 | 0.750 | 47.430 | 48.000 | -0.570 | under -0.6 | 0.840149 | 0.843990 | -0.003841 | 48.000 | -0.003841 | preview-guided backward refinement (spawn=1.00, preview=13.00, backward=0.42, prune=1.24, nfe=47.43) |
| r0004 | 1.000 | 63.554 | 64.000 | -0.446 | under -0.4 | 0.843016 | 0.847290 | -0.004274 | 64.000 | -0.004274 | preview-guided backward refinement (spawn=1.00, preview=18.54, backward=0.45, prune=1.28, nfe=63.55) |
| r0003 | 0.000 | 10.000 | 10.000 | 0.000 | on target | 0.787971 | 0.676462 | 0.111510 | 8.000 | 0.111510 | single-root preview (spawn=1.00, preview=3.00, prune=1.00, nfe=10.00) |
| r0003 | 0.250 | 20.000 | 20.000 | 0.000 | on target | 0.817857 | 0.802083 | 0.015774 | 20.000 | 0.015774 | single-root preview (spawn=1.00, preview=7.00, prune=1.15, nfe=20.00) |
| r0003 | 0.500 | 36.000 | 36.000 | 0.000 | on target | 0.832135 | 0.836650 | -0.004514 | 36.000 | -0.004514 | preview-guided backward refinement (spawn=1.00, preview=10.00, backward=1.00, prune=1.18, nfe=36.00) |
| r0003 | 0.750 | 46.640 | 48.000 | -1.360 | under -1.4 | 0.840608 | 0.843990 | -0.003382 | 48.000 | -0.003382 | preview-guided backward refinement (spawn=1.00, preview=12.82, backward=1.00, prune=1.24, nfe=46.64) |
| r0003 | 1.000 | 63.000 | 64.000 | -1.000 | under -1.0 | 0.843447 | 0.847290 | -0.003843 | 64.000 | -0.003843 | preview-guided backward refinement (spawn=1.00, preview=18.71, backward=1.00, prune=1.28, nfe=63.00) |
| r0002 | 0.000 | 10.000 | 10.000 | 0.000 | on target | 0.787971 | 0.676462 | 0.111510 | 8.000 | 0.111510 | single-root preview (spawn=1.00, preview=3.00, prune=1.00, nfe=10.00) |
| r0002 | 0.250 | 20.000 | 20.000 | 0.000 | on target | 0.801358 | 0.802083 | -0.000725 | 20.000 | -0.000725 | single-root preview (spawn=1.00, preview=6.00, prune=1.00, nfe=20.00) |
| r0002 | 0.500 | 36.000 | 36.000 | 0.000 | on target | 0.836899 | 0.836650 | 0.000250 | 36.000 | 0.000250 | preview-guided backward refinement (spawn=1.00, preview=9.00, backward=1.39, prune=2.00, nfe=36.00) |
| r0002 | 0.750 | 47.000 | 48.000 | -1.000 | under -1.0 | 0.834103 | 0.843990 | -0.009887 | 48.000 | -0.009887 | preview-guided backward refinement (spawn=1.00, preview=11.00, backward=1.33, prune=1.00, nfe=47.00) |
| r0002 | 1.000 | 63.994 | 64.000 | -0.006 | under -0.0 | 0.843780 | 0.847290 | -0.003510 | 64.000 | -0.003510 | preview-guided backward refinement (spawn=1.00, preview=15.29, backward=2.00, prune=1.00, nfe=63.99) |
| r0000 | 0.000 | 10.000 | 10.000 | 0.000 | on target | 0.778068 | 0.676462 | 0.101606 | 8.000 | 0.101606 | single-root preview (spawn=1.00, preview=2.00, prune=1.00, nfe=10.00) |
| r0000 | 0.250 | 20.000 | 20.000 | 0.000 | on target | 0.789670 | 0.802083 | -0.012413 | 20.000 | -0.012413 | single-root preview (spawn=1.00, preview=4.00, prune=1.00, nfe=20.00) |
| r0000 | 0.500 | 36.000 | 36.000 | 0.000 | on target | 0.839043 | 0.836650 | 0.002394 | 36.000 | 0.002394 | single-root preview (spawn=1.00, preview=8.00, prune=2.00, nfe=36.00) |
| r0000 | 0.750 | 48.000 | 48.000 | 0.000 | on target | 0.841569 | 0.843990 | -0.002421 | 48.000 | -0.002421 | single-root preview (spawn=1.00, preview=12.00, prune=2.00, nfe=48.00) |
| r0000 | 1.000 | 64.000 | 64.000 | 0.000 | on target | 0.844569 | 0.847290 | -0.002721 | 64.000 | -0.002721 | single-root preview (spawn=1.00, preview=18.00, prune=2.00, nfe=64.00) |

## Beta Opportunities

Focus first on beta regions that are still below the beta-matched baseline reward.
Use target NFE as a reference for comparability: if a beta is far below the reference compute, that may explain why it still trails baseline.

| beta | latest_round | latest_nfe | target_nfe | latest_reward | latest_vs_beta_target | near_target_best_round | near_target_best_reward | note |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 1.000 | r0004 | 63.554 | 64.000 | 0.843016 | -0.004274 | r0000 | 0.844569 | below reference NFE; likely underusing compute versus baseline |
| 0.750 | r0004 | 47.430 | 48.000 | 0.840149 | -0.003841 | r0000 | 0.841569 | below reference NFE; likely underusing compute versus baseline |
| 0.500 | r0004 | 36.000 | 36.000 | 0.837384 | 0.000735 | r0000 | 0.839043 | already at/above beta-matched baseline |
| 0.250 | r0004 | 20.000 | 20.000 | 0.817857 | 0.015774 | r0004 | 0.817857 | already at/above beta-matched baseline |
| 0.000 | r0004 | 10.000 | 10.000 | 0.787971 | 0.111510 | r0004 | 0.787971 | already at/above beta-matched baseline |

## Regression Ledger

| beta | latest_round | latest_nfe | latest_reward | prev_round | prev_nfe | prev_reward | reward_delta | nfe_delta | beta_target_reward |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.000 | r0004 | 10.000 | 0.787971 | r0003 | 10.000 | 0.787971 | 0.000000 | 0.000 | 0.676462 |
| 0.250 | r0004 | 20.000 | 0.817857 | r0003 | 20.000 | 0.817857 | 0.000000 | 0.000 | 0.802083 |
| 0.500 | r0004 | 36.000 | 0.837384 | r0003 | 36.000 | 0.832135 | 0.005249 | 0.000 | 0.836650 |
| 0.750 | r0004 | 47.430 | 0.840149 | r0003 | 46.640 | 0.840608 | -0.000459 | 0.790 | 0.843990 |
| 1.000 | r0004 | 63.554 | 0.843016 | r0003 | 63.000 | 0.843447 | -0.000431 | 0.554 | 0.847290 |

Use this ledger to avoid repairing one weak beta by silently regressing a previously stronger one.
If a beta already beats or nearly matches its target baseline, prefer protecting it unless the gain elsewhere is clearly larger.

## Rejected Round Lessons

### Rejected `r0005` vs incumbent `r0004`

- rejected round: `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0005_20260527_160800_ffd4e330`
- incumbent reference: `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0004_20260527_160800_ffd4e330`
- rejection reason: candidate did not beat incumbent on fixed-target frontier score

| beta | cand_reward | inc_reward | delta_reward | cand_nfe | inc_nfe | cand_status | main_action_shift | likely lesson |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0.500 | 0.836465 | 0.837384 | -0.000919 | 35.422 | 36.000 | below-target | forward -1.29, preview +0.71, mean_nfe -0.58 | likely too little compute versus the baseline-matched reference |
| 1.000 | 0.842756 | 0.843016 | -0.000261 | 63.588 | 63.554 | below-target | preview +1.63, forward -1.60, backward +0.52 | likely too little compute versus the baseline-matched reference |

Treat these rejected-round notes as negative evidence: avoid repeating the same action-shift pattern unless another beta clearly needs it.

### Rejected `r0001` vs incumbent `r0000`

- rejected round: `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0001_20260527_160800_ffd4e330`
- incumbent reference: `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0000_20260527_160800_ffd4e330`
- rejection reason: candidate did not beat incumbent on fixed-target frontier score

| beta | cand_reward | inc_reward | delta_reward | cand_nfe | inc_nfe | cand_status | main_action_shift | likely lesson |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0.500 | 0.837967 | 0.839043 | -0.001076 | 36.000 | 36.000 | aligned | preview +1.17, prune -0.89 | more preview budget did not translate into better ranking/refinement |
| 0.750 | 0.840621 | 0.841569 | -0.000948 | 48.000 | 48.000 | aligned | preview +2.46, forward +0.89, prune -0.59 | more preview budget did not translate into better ranking/refinement |
| 1.000 | 0.829183 | 0.844569 | -0.015386 | 64.000 | 64.000 | aligned | preview +3.02, backward +1.80 | more preview budget did not translate into better ranking/refinement |

Treat these rejected-round notes as negative evidence: avoid repeating the same action-shift pattern unless another beta clearly needs it.

## Historical Action Effects

Optimization target remains reward-NFE tradeoff; the notes below are hindsight correlations from prior controller changes, not the objective itself.
Use them to understand which action adjustments previously spent more NFE and whether that spend was productive.

| action | when increased | when decreased |
| --- | --- | --- |
| `spawn` | none | none |
| `forward` | 8 cases; mean action +12.50; mean Δreward=+0.001944; mean Δnfe=+0.04 | 4 cases; mean action -2.15; mean Δreward=+0.004477; mean Δnfe=-0.34 |
| `preview` | 7 cases; mean action +1.60; mean Δreward=+0.005336; mean Δnfe=-0.19 | 3 cases; mean action -1.57; mean Δreward=-0.001002; mean Δnfe=-0.34 |
| `backward` | 3 cases; mean action +1.57; mean Δreward=-0.003467; mean Δnfe=-0.34 | 4 cases; mean action -0.78; mean Δreward=+0.001007; mean Δnfe=+0.09 |
| `prune` | none | 3 cases; mean action -0.94; mean Δreward=-0.004340; mean Δnfe=-0.34 |

By-beta action effect summaries:

| beta | action | direction | cases | mean_action_delta | mean_delta_reward | mean_delta_nfe |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0.000 | `forward` | increase | 1 | 5.00 | 0.009903 | 0.000 |
| 0.000 | `preview` | increase | 1 | 1.00 | 0.009903 | 0.000 |
| 0.250 | `forward` | decrease | 1 | -1.00 | 0.016498 | 0.000 |
| 0.250 | `forward` | increase | 1 | 10.00 | 0.011689 | 0.000 |
| 0.250 | `preview` | increase | 2 | 1.50 | 0.014093 | 0.000 |
| 0.500 | `backward` | decrease | 1 | -1.00 | 0.005249 | 0.000 |
| 0.500 | `backward` | increase | 1 | 1.39 | -0.002144 | 0.000 |
| 0.500 | `forward` | decrease | 1 | -1.00 | -0.004764 | 0.000 |
| 0.500 | `forward` | increase | 2 | 10.50 | 0.001552 | 0.000 |
| 0.500 | `preview` | decrease | 1 | -1.00 | 0.005249 | 0.000 |
| 0.500 | `preview` | increase | 2 | 1.00 | -0.003454 | 0.000 |
| 0.500 | `prune` | decrease | 1 | -0.82 | -0.004764 | 0.000 |
| 0.750 | `backward` | decrease | 1 | -0.58 | -0.000459 | 0.790 |
| 0.750 | `backward` | increase | 1 | 1.33 | -0.007466 | -1.000 |
| 0.750 | `forward` | decrease | 1 | -2.18 | 0.006505 | -0.360 |
| 0.750 | `forward` | increase | 2 | 13.81 | -0.003963 | -0.105 |
| 0.750 | `preview` | decrease | 1 | -1.00 | -0.007466 | -1.000 |
| 0.750 | `preview` | increase | 1 | 1.82 | 0.006505 | -0.360 |
| 0.750 | `prune` | decrease | 1 | -1.00 | -0.007466 | -1.000 |
| 1.000 | `backward` | decrease | 2 | -0.77 | -0.000382 | -0.220 |
| 1.000 | `backward` | increase | 1 | 2.00 | -0.000789 | -0.006 |
| 1.000 | `forward` | decrease | 1 | -4.41 | -0.000333 | -0.994 |
| 1.000 | `forward` | increase | 2 | 18.21 | -0.000610 | 0.274 |
| 1.000 | `preview` | decrease | 1 | -2.71 | -0.000789 | -0.006 |
| 1.000 | `preview` | increase | 1 | 3.41 | -0.000333 | -0.994 |
| 1.000 | `prune` | decrease | 1 | -1.00 | -0.000789 | -0.006 |

Recent concrete examples:

| change | beta | action_delta | delta_reward | delta_nfe | note |
| --- | ---: | --- | ---: | ---: | --- |
| r0002->r0003 | 1.000 | forward -4.41 | -0.000333 | -0.994 | preview +3.4, backward -1.0 |
| r0002->r0003 | 1.000 | preview +3.41 | -0.000333 | -0.994 | forward -4.4, backward -1.0 |
| r0002->r0003 | 1.000 | backward -1.00 | -0.000333 | -0.994 | forward -4.4, preview +3.4 |
| r0003->r0004 | 0.500 | forward +1.00 | 0.005249 | 0.000 | preview -1.0, backward -1.0 |
| r0003->r0004 | 0.500 | preview -1.00 | 0.005249 | 0.000 | forward +1.0, backward -1.0 |
| r0003->r0004 | 0.500 | backward -1.00 | 0.005249 | 0.000 | forward +1.0, preview -1.0 |
| r0003->r0004 | 0.750 | forward +0.61 | -0.000459 | 0.790 | backward -0.6 |
| r0003->r0004 | 0.750 | backward -0.58 | -0.000459 | 0.790 | forward +0.6 |
| r0003->r0004 | 1.000 | forward +0.72 | -0.000431 | 0.554 | backward -0.5 |
| r0003->r0004 | 1.000 | backward -0.55 | -0.000431 | 0.554 | forward +0.7 |

## Recent History

### `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0004_20260527_160800_ffd4e330`

- controller snapshot: `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0004_20260527_160800_ffd4e330/flow_autotts/controllers/optimal.py`
- compact summary: `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0004_20260527_160800_ffd4e330/proposal_results/summary.json`

```json
{
  "betas": [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0
  ],
  "budget": 64,
  "evaluated_sample_size": 500,
  "experiment": "pickscore_sd35",
  "num_shards": 4,
  "rounds": [
    {
      "beta_sweep": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 7.0,
            "mean_nfe": 10.0,
            "preview": 3.0,
            "prune": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=3.00, prune=1.00, nfe=10.00)",
          "beta": 0.0,
          "nfe": 10,
          "reward": 0.7879712210893631,
          "reward_per_nfe": 0.07879712210893632
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 13.0,
            "mean_nfe": 20.0,
            "preview": 7.0,
            "prune": 1.148,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=7.00, prune=1.15, nfe=20.00)",
          "beta": 0.25,
          "nfe": 20,
          "reward": 0.8178565890789032,
          "reward_per_nfe": 0.04089282945394516
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 27.0,
            "mean_nfe": 36.0,
            "preview": 9.0,
            "prune": 1.18,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=9.00, prune=1.18, nfe=36.00)",
          "beta": 0.5,
          "nfe": 36,
          "reward": 0.8373842916488647,
          "reward_per_nfe": 0.02326067476802402
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 0.424,
            "forward": 34.434,
            "mean_nfe": 47.43,
            "preview": 12.996,
            "prune": 1.236,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=13.00, backward=0.42, prune=1.24, nfe=47.43)",
          "beta": 0.75,
          "nfe": 47.43,
          "reward": 0.8401492063999176,
          "reward_per_nfe": 0.017715898676385695
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 0.452,
            "forward": 45.014,
            "mean_nfe": 63.554,
            "preview": 18.54,
            "prune": 1.28,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=18.54, backward=0.45, prune=1.28, nfe=63.55)",
          "beta": 1.0,
          "nfe": 63.554,
          "reward": 0.843016294002533,
          "reward_per_nfe": 0.013264788355767018
        }
      ],
      "controller": "optimal",
      "controller_name": "OptimalController",
      "pareto_frontier": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 7.0,
            "mean_nfe": 10.0,
            "preview": 3.0,
            "prune": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=3.00, prune=1.00, nfe=10.00)",
          "beta": 0.0,
          "nfe": 10,
          "reward": 0.7879712210893631,
          "reward_per_nfe": 0.07879712210893632
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 13.0,
            "mean_nfe": 20.0,
            "preview": 7.0,
            "prune": 1.148,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=7.00, prune=1.15, nfe=20.00)",
          "beta": 0.25,
          "nfe": 20,
          "reward": 0.8178565890789032,
          "reward_per_nfe": 0.04089282945394516
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 27.0,
            "mean_nfe": 36.0,
            "preview": 9.0,
            "prune": 1.18,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=9.00, prune=1.18, nfe=36.00)",
          "beta": 0.5,
          "nfe": 36,
          "reward": 0.8373842916488647,
          "reward_per_nfe": 0.02326067476802402
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 0.424,
            "forward": 34.434,
            "mean_nfe": 47.43,
            "preview": 12.996,
            "prune": 1.236,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=13.00, backward=0.42, prune=1.24, nfe=47.43)",
          "beta": 0.75,
          "nfe": 47.43,
          "reward": 0.8401492063999176,
          "reward_per_nfe": 0.017715898676385695
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 0.452,
            "forward": 45.014,
            "mean_nfe": 63.554,
            "preview": 18.54,
            "prune": 1.28,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=18.54, backward=0.45, prune=1.28, nfe=63.55)",
          "beta": 1.0,
          "nfe": 63.554,
          "reward": 0.843016294002533,
          "reward_per_nfe": 0.013264788355767018
        }
      ],
      "round_id": 0
    }
  ],
  "sample_seed": 42,
  "sample_size": 500,
  "shard_index": null
}
```

### `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0003_20260527_160800_ffd4e330`

- controller snapshot: `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0003_20260527_160800_ffd4e330/flow_autotts/controllers/optimal.py`
- compact summary: `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0003_20260527_160800_ffd4e330/proposal_results/summary.json`

```json
{
  "betas": [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0
  ],
  "budget": 64,
  "evaluated_sample_size": 500,
  "experiment": "pickscore_sd35",
  "num_shards": 4,
  "rounds": [
    {
      "beta_sweep": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 7.0,
            "mean_nfe": 10.0,
            "preview": 3.0,
            "prune": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=3.00, prune=1.00, nfe=10.00)",
          "beta": 0.0,
          "nfe": 10,
          "reward": 0.7879712210893631,
          "reward_per_nfe": 0.07879712210893632
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 13.0,
            "mean_nfe": 20.0,
            "preview": 7.0,
            "prune": 1.148,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=7.00, prune=1.15, nfe=20.00)",
          "beta": 0.25,
          "nfe": 20,
          "reward": 0.8178565890789032,
          "reward_per_nfe": 0.04089282945394516
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.0,
            "forward": 26.0,
            "mean_nfe": 36.0,
            "preview": 10.0,
            "prune": 1.18,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=10.00, backward=1.00, prune=1.18, nfe=36.00)",
          "beta": 0.5,
          "nfe": 36,
          "reward": 0.8321353733539582,
          "reward_per_nfe": 0.02311487148205439
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.0,
            "forward": 33.82,
            "mean_nfe": 46.64,
            "preview": 12.82,
            "prune": 1.236,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=12.82, backward=1.00, prune=1.24, nfe=46.64)",
          "beta": 0.75,
          "nfe": 46.64,
          "reward": 0.8406082444190979,
          "reward_per_nfe": 0.01803113442932742
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.0,
            "forward": 44.294,
            "mean_nfe": 63.0,
            "preview": 18.706,
            "prune": 1.28,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=18.71, backward=1.00, prune=1.28, nfe=63.00)",
          "beta": 1.0,
          "nfe": 63,
          "reward": 0.8434469585418701,
          "reward_per_nfe": 0.013388046960982065
        }
      ],
      "controller": "optimal",
      "controller_name": "OptimalController",
      "pareto_frontier": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 7.0,
            "mean_nfe": 10.0,
            "preview": 3.0,
            "prune": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=3.00, prune=1.00, nfe=10.00)",
          "beta": 0.0,
          "nfe": 10,
          "reward": 0.7879712210893631,
          "reward_per_nfe": 0.07879712210893632
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 13.0,
            "mean_nfe": 20.0,
            "preview": 7.0,
            "prune": 1.148,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=7.00, prune=1.15, nfe=20.00)",
          "beta": 0.25,
          "nfe": 20,
          "reward": 0.8178565890789032,
          "reward_per_nfe": 0.04089282945394516
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.0,
            "forward": 26.0,
            "mean_nfe": 36.0,
            "preview": 10.0,
            "prune": 1.18,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=10.00, backward=1.00, prune=1.18, nfe=36.00)",
          "beta": 0.5,
          "nfe": 36,
          "reward": 0.8321353733539582,
          "reward_per_nfe": 0.02311487148205439
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.0,
            "forward": 33.82,
            "mean_nfe": 46.64,
            "preview": 12.82,
            "prune": 1.236,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=12.82, backward=1.00, prune=1.24, nfe=46.64)",
          "beta": 0.75,
          "nfe": 46.64,
          "reward": 0.8406082444190979,
          "reward_per_nfe": 0.01803113442932742
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.0,
            "forward": 44.294,
            "mean_nfe": 63.0,
            "preview": 18.706,
            "prune": 1.28,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=18.71, backward=1.00, prune=1.28, nfe=63.00)",
          "beta": 1.0,
          "nfe": 63,
          "reward": 0.8434469585418701,
          "reward_per_nfe": 0.013388046960982065
        }
      ],
      "round_id": 0
    }
  ],
  "sample_seed": 42,
  "sample_size": 500,
  "shard_index": null
}
```

### `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0002_20260527_160800_ffd4e330`

- controller snapshot: `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0002_20260527_160800_ffd4e330/flow_autotts/controllers/optimal.py`
- compact summary: `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0002_20260527_160800_ffd4e330/proposal_results/summary.json`

```json
{
  "betas": [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0
  ],
  "budget": 64,
  "evaluated_sample_size": 500,
  "experiment": "pickscore_sd35",
  "num_shards": 4,
  "rounds": [
    {
      "beta_sweep": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 7.0,
            "mean_nfe": 10.0,
            "preview": 3.0,
            "prune": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=3.00, prune=1.00, nfe=10.00)",
          "beta": 0.0,
          "nfe": 10,
          "reward": 0.7879712210893631,
          "reward_per_nfe": 0.07879712210893632
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 14.0,
            "mean_nfe": 20.0,
            "preview": 6.0,
            "prune": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=6.00, prune=1.00, nfe=20.00)",
          "beta": 0.25,
          "nfe": 20,
          "reward": 0.8013581433296204,
          "reward_per_nfe": 0.040067907166481016
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.39,
            "forward": 27.0,
            "mean_nfe": 36.0,
            "preview": 9.0,
            "prune": 2.0,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=9.00, backward=1.39, prune=2.00, nfe=36.00)",
          "beta": 0.5,
          "nfe": 36,
          "reward": 0.8368991909027099,
          "reward_per_nfe": 0.023247199747297498
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.332,
            "forward": 36.0,
            "mean_nfe": 47.0,
            "preview": 11.0,
            "prune": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=11.00, backward=1.33, prune=1.00, nfe=47.00)",
          "beta": 0.75,
          "nfe": 47,
          "reward": 0.8341028877496719,
          "reward_per_nfe": 0.01774686995212068
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 2.0,
            "forward": 48.702,
            "mean_nfe": 63.994,
            "preview": 15.292,
            "prune": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=15.29, backward=2.00, prune=1.00, nfe=63.99)",
          "beta": 1.0,
          "nfe": 63.994,
          "reward": 0.8437798924446106,
          "reward_per_nfe": 0.013185457155699768
        }
      ],
      "controller": "optimal",
      "controller_name": "OptimalController",
      "pareto_frontier": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 7.0,
            "mean_nfe": 10.0,
            "preview": 3.0,
            "prune": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=3.00, prune=1.00, nfe=10.00)",
          "beta": 0.0,
          "nfe": 10,
          "reward": 0.7879712210893631,
          "reward_per_nfe": 0.07879712210893632
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 14.0,
            "mean_nfe": 20.0,
            "preview": 6.0,
            "prune": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=6.00, prune=1.00, nfe=20.00)",
          "beta": 0.25,
          "nfe": 20,
          "reward": 0.8013581433296204,
          "reward_per_nfe": 0.040067907166481016
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 1.39,
            "forward": 27.0,
            "mean_nfe": 36.0,
            "preview": 9.0,
            "prune": 2.0,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=9.00, backward=1.39, prune=2.00, nfe=36.00)",
          "beta": 0.5,
          "nfe": 36,
          "reward": 0.8368991909027099,
          "reward_per_nfe": 0.023247199747297498
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "backward": 2.0,
            "forward": 48.702,
            "mean_nfe": 63.994,
            "preview": 15.292,
            "prune": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "preview-guided backward refinement (spawn=1.00, preview=15.29, backward=2.00, prune=1.00, nfe=63.99)",
          "beta": 1.0,
          "nfe": 63.994,
          "reward": 0.8437798924446106,
          "reward_per_nfe": 0.013185457155699768
        }
      ],
      "round_id": 0
    }
  ],
  "sample_seed": 42,
  "sample_size": 500,
  "shard_index": null
}
```

### `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0000_20260527_160800_ffd4e330`

- controller snapshot: `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0000_20260527_160800_ffd4e330/flow_autotts/controllers/optimal.py`
- compact summary: `logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0000_20260527_160800_ffd4e330/proposal_results/summary.json`

```json
{
  "betas": [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0
  ],
  "budget": 64,
  "evaluated_sample_size": 500,
  "experiment": "pickscore_sd35",
  "num_shards": 4,
  "rounds": [
    {
      "beta_sweep": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 2.0,
            "mean_nfe": 10.0,
            "preview": 2.0,
            "prune": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=2.00, prune=1.00, nfe=10.00)",
          "beta": 0.0,
          "nfe": 10,
          "reward": 0.7780677300691604,
          "reward_per_nfe": 0.07780677300691605
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 4.0,
            "mean_nfe": 20.0,
            "preview": 4.0,
            "prune": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=4.00, prune=1.00, nfe=20.00)",
          "beta": 0.25,
          "nfe": 20,
          "reward": 0.7896696199178695,
          "reward_per_nfe": 0.03948348099589348
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 7.0,
            "mean_nfe": 36.0,
            "preview": 8.0,
            "prune": 2.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=8.00, prune=2.00, nfe=36.00)",
          "beta": 0.5,
          "nfe": 36,
          "reward": 0.8390433425903321,
          "reward_per_nfe": 0.02330675951639811
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 9.0,
            "mean_nfe": 48.0,
            "preview": 12.0,
            "prune": 2.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=12.00, prune=2.00, nfe=48.00)",
          "beta": 0.75,
          "nfe": 48,
          "reward": 0.8415692585706711,
          "reward_per_nfe": 0.01753269288688898
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 13.0,
            "mean_nfe": 64.0,
            "preview": 18.0,
            "prune": 2.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=18.00, prune=2.00, nfe=64.00)",
          "beta": 1.0,
          "nfe": 64,
          "reward": 0.8445690048933029,
          "reward_per_nfe": 0.013196390701457858
        }
      ],
      "controller": "optimal",
      "controller_name": "OptimalController",
      "pareto_frontier": [
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 2.0,
            "mean_nfe": 10.0,
            "preview": 2.0,
            "prune": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=2.00, prune=1.00, nfe=10.00)",
          "beta": 0.0,
          "nfe": 10,
          "reward": 0.7780677300691604,
          "reward_per_nfe": 0.07780677300691605
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 4.0,
            "mean_nfe": 20.0,
            "preview": 4.0,
            "prune": 1.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=4.00, prune=1.00, nfe=20.00)",
          "beta": 0.25,
          "nfe": 20,
          "reward": 0.7896696199178695,
          "reward_per_nfe": 0.03948348099589348
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 7.0,
            "mean_nfe": 36.0,
            "preview": 8.0,
            "prune": 2.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=8.00, prune=2.00, nfe=36.00)",
          "beta": 0.5,
          "nfe": 36,
          "reward": 0.8390433425903321,
          "reward_per_nfe": 0.02330675951639811
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 9.0,
            "mean_nfe": 48.0,
            "preview": 12.0,
            "prune": 2.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=12.00, prune=2.00, nfe=48.00)",
          "beta": 0.75,
          "nfe": 48,
          "reward": 0.8415692585706711,
          "reward_per_nfe": 0.01753269288688898
        },
        {
          "action_statistics": {
            "answer": 1.0,
            "forward": 13.0,
            "mean_nfe": 64.0,
            "preview": 18.0,
            "prune": 2.0,
            "spawn": 1.0
          },
          "behavior_summary": "single-root preview (spawn=1.00, preview=18.00, prune=2.00, nfe=64.00)",
          "beta": 1.0,
          "nfe": 64,
          "reward": 0.8445690048933029,
          "reward_per_nfe": 0.013196390701457858
        }
      ],
      "round_id": 0
    }
  ],
  "sample_seed": 42,
  "sample_size": 500,
  "shard_index": null
}
```

