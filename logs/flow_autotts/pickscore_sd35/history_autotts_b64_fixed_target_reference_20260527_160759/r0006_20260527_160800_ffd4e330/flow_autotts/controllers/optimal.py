"""Candidate controller for SD3.5 PickScore discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Beta-scheduled staged search with reserved late tie-breaking."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        schedule = self._schedule(beta, int(env.budget))
        initial_budget = int(env.budget_left)

        if bool(schedule["single_root"]):
            return self._single_root_preview(env, schedule)

        try:
            root_ids = env.spawn(int(schedule["roots"]))
        except InvalidActionError:
            return self._safe_answer(env)

        scout_ids: list[int] = []
        for index, particle_id in enumerate(root_ids):
            scout_cost = self._segment_cost(env, particle_id, float(schedule["scout_time"])) + 1
            if not self._can_spend(env, initial_budget, schedule, scout_cost):
                break
            solver = "sde" if index < int(schedule["sde_scouts"]) else "euler"
            cfg = self._sde_cfg(schedule) if solver == "sde" else None
            try:
                self._forward_to(env, particle_id, float(schedule["scout_time"]), solver=solver, cfg=cfg)
                self._preview(env, particle_id)
                scout_ids.append(particle_id)
            except (BudgetExceededError, InvalidActionError):
                break

        if not scout_ids:
            return self._finish_one(env, root_ids[0] if root_ids else None)

        scout_survivors = self._adaptive_keep(
            env,
            scout_ids,
            int(schedule["keep_after_scout"]),
            float(schedule["scout_margin"]),
        )
        self._prune_non_survivors(env, root_ids, scout_survivors)

        try:
            finalists = self._commit_stage(env, scout_survivors, schedule, initial_budget)
            self._rival_stage(env, finalists, schedule, initial_budget)
            finalists = self._repair_stage(env, finalists, schedule, initial_budget)
            self._tail_stage(env, finalists, schedule, initial_budget)
            self._late_confirmation_stage(env, finalists, schedule, initial_budget)
            self._fill_target(env, finalists, schedule, initial_budget)
        except BudgetExceededError:
            return self._safe_answer(env)

        return self._safe_answer(env)

    def _schedule(self, beta: float, budget: int) -> dict[str, float | int | bool | str]:
        target_nfe = self._target_nfe(budget, beta)
        if beta <= 0.0:
            return {
                "target_nfe": target_nfe,
                "single_root": True,
                "preview_goal": 3,
                "single_root_fill": False,
            }
        if beta <= 0.25:
            return {
                "target_nfe": target_nfe,
                "single_root": True,
                "preview_goal": 7,
                "single_root_fill": False,
            }
        if beta <= 0.5:
            return {
                "target_nfe": target_nfe,
                "single_root": False,
                "roots": 4,
                "sde_scouts": 0,
                "sde_noise_scale": 0.0,
                "scout_time": 0.3,
                "commit_time": 0.7,
                "rival_time": 0.9,
                "repair_time": 0.7,
                "repair_eval_time": 0.9,
                "keep_after_scout": 2,
                "keep_after_commit": 2,
                "scout_margin": 0.036,
                "commit_margin": 0.030,
                "rival_gap": 0.024,
                "repair_gap": 0.010,
                "uncertainty_gate": 0.18,
                "max_rival_rounds": 1,
                "max_repair_anchors": 1,
                "children_per_anchor": 1,
                "noise_policy": "fresh_noise",
                "strength": 1.0,
                "tail_top_time": 1.0,
                "tail_rival_time": 1.0,
                "tail_rival_gap": 0.014,
                "late_confirmation_rounds": 0,
                "late_confirmation_gap": 0.012,
                "late_confirmation_uncertainty": 0.14,
                "late_confirmation_margin": 0.010,
                "fill_confirmation_cap": 1,
                "fill_gap": 0.016,
                "fill_uncertainty_gate": 0.14,
                "late_repair_children": 0,
                "late_repair_time": 0.9,
                "late_repair_eval_time": 1.0,
                "reserve_nfe": 0,
                "stop_gap": 0.020,
                "stop_spend_floor": 35,
            }
        if beta <= 0.75:
            return {
                "target_nfe": target_nfe,
                "single_root": False,
                "roots": 5,
                "sde_scouts": 1,
                "sde_noise_scale": 0.010,
                "scout_time": 0.3,
                "commit_time": 0.7,
                "rival_time": 0.9,
                "repair_time": 0.7,
                "repair_eval_time": 0.9,
                "keep_after_scout": 3,
                "keep_after_commit": 3,
                "scout_margin": 0.032,
                "commit_margin": 0.026,
                "rival_gap": 0.020,
                "repair_gap": 0.010,
                "uncertainty_gate": 0.17,
                "max_rival_rounds": 2,
                "max_repair_anchors": 1,
                "children_per_anchor": 1,
                "noise_policy": "fresh_noise",
                "strength": 0.85,
                "tail_top_time": 1.0,
                "tail_rival_time": 1.0,
                "tail_rival_gap": 0.011,
                "late_confirmation_rounds": 1,
                "late_confirmation_gap": 0.015,
                "late_confirmation_uncertainty": 0.135,
                "late_confirmation_margin": 0.010,
                "fill_confirmation_cap": 2,
                "fill_gap": 0.014,
                "fill_uncertainty_gate": 0.13,
                "late_repair_children": 0,
                "late_repair_time": 0.9,
                "late_repair_eval_time": 1.0,
                "reserve_nfe": 4,
                "stop_gap": 0.017,
                "stop_spend_floor": 46,
            }
        return {
            "target_nfe": min(target_nfe, 64),
            "single_root": False,
            "roots": 6,
            "sde_scouts": 1,
            "sde_noise_scale": 0.008,
            "scout_time": 0.3,
            "commit_time": 0.7,
            "rival_time": 0.9,
            "repair_time": 0.7,
            "repair_eval_time": 0.9,
            "keep_after_scout": 4,
            "keep_after_commit": 4,
            "scout_margin": 0.028,
            "commit_margin": 0.023,
            "rival_gap": 0.017,
            "repair_gap": 0.009,
            "uncertainty_gate": 0.16,
            "max_rival_rounds": 2,
            "max_repair_anchors": 1,
            "children_per_anchor": 1,
            "noise_policy": "mixed_noise",
            "strength": 0.72,
            "tail_top_time": 1.0,
            "tail_rival_time": 1.0,
            "tail_rival_gap": 0.010,
            "late_confirmation_rounds": 2,
            "late_confirmation_gap": 0.014,
            "late_confirmation_uncertainty": 0.13,
            "late_confirmation_margin": 0.010,
            "fill_confirmation_cap": 2,
            "fill_gap": 0.013,
            "fill_uncertainty_gate": 0.125,
            "late_repair_children": 1,
            "late_repair_time": 0.9,
            "late_repair_eval_time": 1.0,
            "reserve_nfe": 6,
            "stop_gap": 0.015,
            "stop_spend_floor": 61,
        }

    def _target_nfe(self, budget: int, beta: float) -> int:
        budget = max(0, int(budget))
        if beta <= 0.0:
            return min(budget, 10)
        if beta <= 0.25:
            return min(budget, 20)
        if beta <= 0.5:
            return min(budget, 36)
        if beta <= 0.75:
            return min(budget, 48)
        return min(budget, 64)

    def _single_root_preview(
        self,
        env: FlowTTSEnv,
        schedule: dict[str, float | int | bool | str],
    ) -> AnswerRecord:
        try:
            root_id = env.spawn(1)[0]
        except InvalidActionError:
            return self._safe_answer(env)

        try:
            target_nfe = int(schedule["target_nfe"])
            preview_goal = max(1, int(schedule["preview_goal"]))
            max_steps = max(0, len(env.time_grid) - 1)
            forward_steps = min(max_steps, max(1, target_nfe - preview_goal))
            preview_count = max(1, target_nfe - forward_steps)
            preview_steps = self._spread_preview_steps(forward_steps, preview_count)

            for step_index in range(1, forward_steps + 1):
                self._forward_to(env, root_id, float(env.time_grid[step_index]), solver="euler")
                if step_index in preview_steps:
                    self._preview(env, root_id)

            while env.nfe_used < target_nfe and env.budget_left > 0 and self._can_repeat_preview(env, root_id):
                self._preview(env, root_id)
        except (BudgetExceededError, InvalidActionError):
            return self._safe_answer(env)

        return self._safe_answer(env)

    def _spread_preview_steps(self, forward_steps: int, preview_count: int) -> set[int]:
        if forward_steps <= 0 or preview_count <= 0:
            return set()
        return {
            max(1, min(forward_steps, int(round(index * forward_steps / preview_count))))
            for index in range(1, preview_count + 1)
        }

    def _commit_stage(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | bool | str],
        initial_budget: int,
    ) -> list[int]:
        ranked = self._rank_particle_ids(env, particle_ids)
        committed: list[int] = []

        for particle_id in ranked[: max(1, int(schedule["keep_after_commit"]))]:
            cost = self._segment_cost(env, particle_id, float(schedule["commit_time"])) + 1
            if not self._can_spend(env, initial_budget, schedule, cost):
                break
            try:
                self._forward_to(env, particle_id, float(schedule["commit_time"]), solver="euler")
                self._preview(env, particle_id)
                committed.append(particle_id)
            except (BudgetExceededError, InvalidActionError):
                break

        if not committed:
            return particle_ids[:1]

        survivors = self._adaptive_keep(
            env,
            committed,
            int(schedule["keep_after_commit"]),
            float(schedule["commit_margin"]),
        )
        self._prune_non_survivors(env, particle_ids, survivors)
        return survivors

    def _rival_stage(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | bool | str],
        initial_budget: int,
    ) -> None:
        if len(particle_ids) < 2:
            return

        rounds = 0
        while rounds < int(schedule["max_rival_rounds"]):
            previews = self._rank_previews(env, particle_ids)
            if len(previews) < 2:
                return
            best = previews[0]
            runner_up = previews[1]
            gap = float(best.score or 0.0) - float(runner_up.score or 0.0)
            uncertainty = max(float(best.uncertainty or 0.0), float(runner_up.uncertainty or 0.0))
            if gap > float(schedule["rival_gap"]) and uncertainty < float(schedule["uncertainty_gate"]):
                return

            progressed = False
            for preview in previews[:2]:
                state = env.get_state()
                particle = state.particles.get(preview.particle_id)
                if particle is None or particle.status != "active":
                    continue
                if particle.time + 1e-9 >= float(schedule["rival_time"]):
                    continue
                cost = self._segment_cost(env, preview.particle_id, float(schedule["rival_time"])) + 1
                if not self._can_spend(env, initial_budget, schedule, cost):
                    continue
                try:
                    self._forward_to(env, preview.particle_id, float(schedule["rival_time"]), solver="euler")
                    self._preview(env, preview.particle_id)
                    progressed = True
                except (BudgetExceededError, InvalidActionError):
                    continue
            if not progressed:
                return
            rounds += 1

    def _repair_stage(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | bool | str],
        initial_budget: int,
    ) -> list[int]:
        previews = self._rank_previews(env, particle_ids)
        if len(previews) < 2 or int(schedule["max_repair_anchors"]) <= 0:
            return particle_ids

        best = previews[0]
        runner_up = previews[1]
        gap = float(best.score or 0.0) - float(runner_up.score or 0.0)
        uncertainty = max(float(best.uncertainty or 0.0), float(runner_up.uncertainty or 0.0))
        if gap > float(schedule["repair_gap"]) and uncertainty < float(schedule["uncertainty_gate"]):
            return particle_ids

        anchor_candidates = [runner_up, best]
        child_cost = self._child_eval_cost(
            env,
            float(schedule["repair_time"]),
            float(schedule["repair_eval_time"]),
        )
        new_ids: list[int] = []

        for anchor in anchor_candidates[: int(schedule["max_repair_anchors"])]:
            projected_cost = int(schedule["children_per_anchor"]) * child_cost
            if not self._can_spend(env, initial_budget, schedule, projected_cost):
                break
            try:
                child_ids = env.backward(
                    anchor.id,
                    target_time=float(schedule["repair_time"]),
                    noise_policy=str(schedule["noise_policy"]),
                    num_children=int(schedule["children_per_anchor"]),
                    strength=float(schedule["strength"]),
                )
            except (BudgetExceededError, InvalidActionError):
                continue

            for child_id in child_ids:
                try:
                    self._forward_to(env, child_id, float(schedule["repair_eval_time"]), solver="euler")
                    self._preview(env, child_id)
                    new_ids.append(child_id)
                except (BudgetExceededError, InvalidActionError):
                    break

        merged = particle_ids + new_ids
        if not new_ids:
            return merged

        survivors = self._adaptive_keep(
            env,
            merged,
            int(schedule["keep_after_commit"]),
            float(schedule["commit_margin"]),
        )
        self._prune_non_survivors(env, merged, survivors)
        return survivors

    def _tail_stage(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | bool | str],
        initial_budget: int,
    ) -> None:
        ranked = self._rank_particle_ids(env, particle_ids)
        if not ranked:
            return

        top_id = ranked[0]
        top_cost = self._segment_cost(env, top_id, float(schedule["tail_top_time"])) + 1
        if self._can_spend(env, initial_budget, schedule, top_cost):
            try:
                self._forward_to(env, top_id, float(schedule["tail_top_time"]), solver="euler")
                self._preview(env, top_id)
            except (BudgetExceededError, InvalidActionError):
                return

        if len(ranked) < 2:
            return

        updated = self._rank_previews(env, particle_ids)
        if len(updated) < 2:
            return
        best = updated[0]
        runner_up = updated[1]
        gap = float(best.score or 0.0) - float(runner_up.score or 0.0)
        uncertainty = max(float(best.uncertainty or 0.0), float(runner_up.uncertainty or 0.0))
        if gap > float(schedule["tail_rival_gap"]) and uncertainty < float(schedule["uncertainty_gate"]):
            return

        rival_id = runner_up.particle_id
        rival_cost = self._segment_cost(env, rival_id, float(schedule["tail_rival_time"])) + 1
        if not self._can_spend(env, initial_budget, schedule, rival_cost):
            return
        try:
            self._forward_to(env, rival_id, float(schedule["tail_rival_time"]), solver="euler")
            self._preview(env, rival_id)
        except (BudgetExceededError, InvalidActionError):
            return

    def _late_confirmation_stage(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | bool | str],
        initial_budget: int,
    ) -> None:
        rounds = int(schedule["late_confirmation_rounds"])
        if rounds <= 0:
            return

        for _ in range(rounds):
            previews = self._rank_previews(env, particle_ids)
            if len(previews) < 2:
                return
            best = previews[0]
            runner_up = previews[1]
            gap = float(best.score or 0.0) - float(runner_up.score or 0.0)
            uncertainty = max(float(best.uncertainty or 0.0), float(runner_up.uncertainty or 0.0))
            if gap > float(schedule["late_confirmation_gap"]) and uncertainty < float(
                schedule["late_confirmation_uncertainty"]
            ):
                return

            candidate_ids = self._close_rivals(
                env,
                particle_ids,
                margin=float(schedule["late_confirmation_margin"]),
            )
            progressed = False
            for particle_id in candidate_ids[:2]:
                next_time = self._next_time_for_particle(env, particle_id)
                if next_time is None:
                    continue
                cost = self._segment_cost(env, particle_id, next_time) + 1
                if not self._can_spend(env, initial_budget, schedule, cost, allow_reserve=True):
                    continue
                try:
                    self._forward_to(env, particle_id, next_time, solver="euler")
                    self._preview(env, particle_id)
                    progressed = True
                except (BudgetExceededError, InvalidActionError):
                    continue
            if not progressed:
                return

    def _fill_target(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | bool | str],
        initial_budget: int,
    ) -> None:
        while True:
            previews = self._rank_previews(env, particle_ids)
            if len(previews) >= 2:
                gap = float(previews[0].score or 0.0) - float(previews[1].score or 0.0)
                uncertainty = max(
                    float(previews[0].uncertainty or 0.0),
                    float(previews[1].uncertainty or 0.0),
                )
                spent = self._spent(env, initial_budget)
                if (
                    spent >= int(schedule["stop_spend_floor"])
                    and gap > float(schedule["stop_gap"])
                    and uncertainty < float(schedule["fill_uncertainty_gate"])
                ):
                    return

                if gap <= float(schedule["fill_gap"]) or uncertainty >= float(schedule["fill_uncertainty_gate"]):
                    progressed = False
                    for particle_id in self._close_rivals(
                        env,
                        particle_ids,
                        margin=float(schedule["late_confirmation_margin"]),
                    )[: int(schedule["fill_confirmation_cap"])]:
                        next_time = self._next_time_for_particle(env, particle_id)
                        if next_time is None:
                            continue
                        cost = self._segment_cost(env, particle_id, next_time) + 1
                        if not self._can_spend(env, initial_budget, schedule, cost, allow_reserve=True):
                            continue
                        try:
                            self._forward_to(env, particle_id, next_time, solver="euler")
                            self._preview(env, particle_id)
                            progressed = True
                        except (BudgetExceededError, InvalidActionError):
                            continue
                    if progressed:
                        continue

            candidate_id, next_time = self._best_next_step(env, particle_ids)
            if candidate_id is None or next_time is None:
                self._late_repair_fill(env, particle_ids, schedule, initial_budget)
                return
            cost = self._segment_cost(env, candidate_id, next_time) + 1
            if not self._can_spend(env, initial_budget, schedule, cost, allow_reserve=True):
                return
            try:
                self._forward_to(env, candidate_id, next_time, solver="euler")
                self._preview(env, candidate_id)
            except (BudgetExceededError, InvalidActionError):
                return

    def _late_repair_fill(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | bool | str],
        initial_budget: int,
    ) -> None:
        if int(schedule.get("late_repair_children", 0)) <= 0:
            return

        previews = self._rank_previews(env, particle_ids)
        if not previews:
            return
        if len(previews) >= 2:
            gap = float(previews[0].score or 0.0) - float(previews[1].score or 0.0)
            uncertainty = max(
                float(previews[0].uncertainty or 0.0),
                float(previews[1].uncertainty or 0.0),
            )
            if gap > float(schedule["fill_gap"]) and uncertainty < float(schedule["fill_uncertainty_gate"]):
                return

        anchor = previews[1] if len(previews) >= 2 else previews[0]
        cost = int(schedule["late_repair_children"]) * self._child_eval_cost(
            env,
            float(schedule["late_repair_time"]),
            float(schedule["late_repair_eval_time"]),
        )
        if not self._can_spend(env, initial_budget, schedule, cost, allow_reserve=True):
            return

        try:
            child_ids = env.backward(
                anchor.id,
                target_time=float(schedule["late_repair_time"]),
                noise_policy=str(schedule["noise_policy"]),
                num_children=int(schedule["late_repair_children"]),
                strength=float(schedule["strength"]),
            )
        except (BudgetExceededError, InvalidActionError):
            return

        for child_id in child_ids:
            try:
                self._forward_to(env, child_id, float(schedule["late_repair_eval_time"]), solver="euler")
                self._preview(env, child_id)
            except (BudgetExceededError, InvalidActionError):
                return

    def _adaptive_keep(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        base_keep: int,
        margin: float,
    ) -> list[int]:
        ranked_previews = self._rank_previews(env, particle_ids)
        if not ranked_previews:
            return particle_ids[: max(1, base_keep)]

        keep = max(1, min(base_keep, len(ranked_previews)))
        cutoff = float(ranked_previews[keep - 1].score or 0.0)
        survivors: list[int] = []
        for preview in ranked_previews:
            score = float(preview.score or 0.0)
            if len(survivors) < keep or score >= cutoff - margin:
                survivors.append(preview.particle_id)
        return survivors

    def _rank_particle_ids(self, env: FlowTTSEnv, particle_ids: list[int]) -> list[int]:
        previews = self._rank_previews(env, particle_ids)
        ranked = [preview.particle_id for preview in previews]
        seen = set(ranked)
        state = env.get_state()
        fallback = [
            particle_id
            for particle_id in particle_ids
            if particle_id in state.particles
            and state.particles[particle_id].status == "active"
            and particle_id not in seen
        ]
        fallback.sort(
            key=lambda particle_id: (
                float(state.particles[particle_id].time),
                -int(particle_id),
            ),
            reverse=True,
        )
        return ranked + fallback

    def _rank_previews(self, env: FlowTTSEnv, particle_ids: list[int]) -> list[PreviewRecord]:
        state = env.get_state()
        previews: list[PreviewRecord] = []
        for particle_id in particle_ids:
            particle = state.particles.get(particle_id)
            if particle is None or particle.status == "pruned" or particle.last_preview_id is None:
                continue
            preview = state.previews.get(particle.last_preview_id)
            if preview is not None and preview.score is not None:
                previews.append(preview)
        return sorted(
            previews,
            key=lambda preview: (
                float(preview.score or 0.0),
                -float(preview.uncertainty or 0.0),
                float(preview.time),
                -preview.id,
            ),
            reverse=True,
        )

    def _close_rivals(self, env: FlowTTSEnv, particle_ids: list[int], margin: float) -> list[int]:
        previews = self._rank_previews(env, particle_ids)
        if not previews:
            return []
        top_score = float(previews[0].score or 0.0)
        close = [
            preview.particle_id
            for preview in previews
            if top_score - float(preview.score or 0.0) <= margin
        ]
        if len(close) >= 2:
            return close
        ranked = [preview.particle_id for preview in previews[:2]]
        return ranked if ranked else close

    def _best_next_step(self, env: FlowTTSEnv, particle_ids: list[int]) -> tuple[int | None, float | None]:
        state = env.get_state()
        ranked_ids = self._rank_particle_ids(env, particle_ids)
        previews = self._rank_previews(env, particle_ids)

        if len(previews) >= 2:
            best = previews[0]
            runner_up = previews[1]
            uncertain = max(float(best.uncertainty or 0.0), float(runner_up.uncertainty or 0.0))
            gap = float(best.score or 0.0) - float(runner_up.score or 0.0)
            if gap < 0.012 or uncertain > 0.14:
                ranked_ids = [runner_up.particle_id, best.particle_id] + [
                    particle_id
                    for particle_id in ranked_ids
                    if particle_id not in {best.particle_id, runner_up.particle_id}
                ]

        for particle_id in ranked_ids:
            particle = state.particles.get(particle_id)
            if particle is None or particle.status != "active":
                continue
            for time_value in env.time_grid:
                if time_value > particle.time + 1e-9:
                    return particle_id, float(time_value)
        return None, None

    def _next_time_for_particle(self, env: FlowTTSEnv, particle_id: int) -> float | None:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None or particle.status != "active":
            return None
        for time_value in env.time_grid:
            if time_value > particle.time + 1e-9:
                return float(time_value)
        return None

    def _segment_cost(self, env: FlowTTSEnv, particle_id: int, target_time: float) -> int:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None:
            return 0
        return sum(1 for time_value in env.time_grid if particle.time < time_value <= target_time + 1e-9)

    def _child_eval_cost(self, env: FlowTTSEnv, child_time: float, eval_time: float) -> int:
        return sum(1 for time_value in env.time_grid if child_time < time_value <= eval_time + 1e-9) + 1

    def _can_spend(
        self,
        env: FlowTTSEnv,
        initial_budget: int,
        schedule: dict[str, float | int | bool | str],
        extra_cost: int,
        allow_reserve: bool = False,
    ) -> bool:
        if extra_cost < 0:
            return False
        spent = self._spent(env, initial_budget)
        limit = min(int(schedule["target_nfe"]), int(env.budget))
        reserve = max(0, int(schedule.get("reserve_nfe", 0)))
        if reserve > 0 and spent < limit - reserve and not allow_reserve:
            effective_limit = limit - reserve
        else:
            effective_limit = limit
        return env.budget_left >= extra_cost and spent + extra_cost <= effective_limit

    def _spent(self, env: FlowTTSEnv, initial_budget: int) -> int:
        return max(0, int(initial_budget - env.budget_left))

    def _forward_to(
        self,
        env: FlowTTSEnv,
        particle_id: int,
        target_time: float,
        solver: str,
        cfg: dict[str, float | str] | None = None,
    ) -> None:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None or particle.status != "active":
            raise InvalidActionError("particle unavailable for forward")
        for time_value in env.time_grid:
            if particle.time < time_value <= target_time + 1e-9:
                env.forward(particle_id, target_time=float(time_value), solver=solver, cfg=cfg)
                particle = env.get_state().particles[particle_id]
                if particle.status != "active":
                    break

    def _preview(self, env: FlowTTSEnv, particle_id: int) -> PreviewRecord:
        return env.preview(particle_id, mode="clean_anchor", scorer="default")

    def _can_repeat_preview(self, env: FlowTTSEnv, particle_id: int) -> bool:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None or particle.last_preview_id is None:
            return True
        preview = state.previews.get(particle.last_preview_id)
        return preview is None or preview.time + 1e-9 < particle.time

    def _prune_non_survivors(self, env: FlowTTSEnv, all_ids: list[int], survivor_ids: list[int]) -> None:
        survivor_set = set(survivor_ids)
        state = env.get_state()
        to_prune = [
            particle_id
            for particle_id in all_ids
            if particle_id in state.particles
            and state.particles[particle_id].status == "active"
            and particle_id not in survivor_set
        ]
        if not to_prune:
            return
        try:
            env.prune(to_prune)
        except InvalidActionError:
            return

    def _finish_one(self, env: FlowTTSEnv, particle_id: int | None) -> AnswerRecord:
        if particle_id is None:
            return self._safe_answer(env)
        try:
            self._forward_to(env, particle_id, 1.0, solver="euler")
            self._preview(env, particle_id)
        except (BudgetExceededError, InvalidActionError):
            pass
        return self._safe_answer(env)

    def _safe_answer(self, env: FlowTTSEnv) -> AnswerRecord:
        state = env.get_state()
        if state.previews:
            return env.answer(rule="best_preview_score")
        return env.answer(rule="latest_active")

    def _sde_cfg(self, schedule: dict[str, float | int | bool | str]) -> dict[str, float | str]:
        return {
            "noise_scale": float(schedule["sde_noise_scale"]),
            "sigma_max": 1.25,
            "min_time": 0.02,
        }
