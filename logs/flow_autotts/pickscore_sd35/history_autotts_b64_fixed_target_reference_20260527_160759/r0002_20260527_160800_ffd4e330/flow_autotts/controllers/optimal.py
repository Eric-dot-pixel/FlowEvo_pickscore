"""Candidate controller for SD3.5 PickScore discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Beta-scheduled scout ladder with ambiguity-triggered local repair."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        schedule = self._schedule(beta, int(env.budget))
        initial_budget = int(env.budget_left)

        try:
            root_ids = env.spawn(int(schedule["roots"]))
        except InvalidActionError:
            return self._safe_answer(env)

        scout_ids: list[int] = []
        sde_scouts = int(schedule["sde_scouts"])
        sde_start = max(0, len(root_ids) - sde_scouts)

        for index, particle_id in enumerate(root_ids):
            scout_cost = self._segment_cost(env, particle_id, float(schedule["scout_time"])) + 1
            if not self._can_spend(env, initial_budget, schedule, scout_cost):
                break
            solver = "sde" if index >= sde_start and sde_scouts > 0 else "euler"
            cfg = None
            if solver == "sde":
                cfg = {
                    "noise_scale": float(schedule["sde_noise_scale"]),
                    "sigma_max": 1.25,
                    "min_time": 0.02,
                }
            try:
                self._forward_to(
                    env,
                    particle_id,
                    float(schedule["scout_time"]),
                    solver=solver,
                    cfg=cfg,
                )
                self._preview(env, particle_id)
                scout_ids.append(particle_id)
            except (BudgetExceededError, InvalidActionError):
                break

        if not scout_ids:
            return self._finish_one(env, root_ids[0] if root_ids else None)

        ranked_scouts = self._rank_particle_ids_from_state(env, scout_ids)
        scout_survivors = ranked_scouts[: max(1, int(schedule["keep_after_scout"]))]
        self._prune_non_survivors(env, scout_ids, scout_survivors)

        try:
            finalists = self._commit_stage(env, scout_survivors, schedule, initial_budget)
            finalists = self._refine_stage(env, finalists, schedule, initial_budget)
            self._tail_stage(env, finalists, schedule, initial_budget)
            self._fill_with_late_steps(env, finalists, schedule, initial_budget)
        except BudgetExceededError:
            return self._safe_answer(env)

        return self._safe_answer(env)

    def _schedule(self, beta: float, budget: int) -> dict[str, float | int | tuple[float, ...] | bool]:
        target_nfe = self._target_nfe(budget, beta)
        if beta <= 0.0:
            return {
                "target_nfe": target_nfe,
                "roots": 2,
                "scout_time": 0.2,
                "keep_after_scout": 1,
                "commit_count": 1,
                "commit_time": 0.5,
                "decisive_time": 0.5,
                "keep_after_commit": 1,
                "prune_margin": 0.06,
                "decisive_gap": 0.05,
                "refine_gap": 0.0,
                "uncertainty_gate": 0.22,
                "max_children": 0,
                "max_refine_anchors": 0,
                "refine_time": 0.8,
                "refine_eval_time": 0.9,
                "tail_targets": (),
                "noise_policy": "inferred_noise",
                "strength": 0.2,
                "sde_scouts": 0,
                "sde_noise_scale": 0.0,
            }
        if beta <= 0.25:
            return {
                "target_nfe": target_nfe,
                "roots": 4,
                "scout_time": 0.2,
                "keep_after_scout": 2,
                "commit_count": 2,
                "commit_time": 0.5,
                "decisive_time": 0.9,
                "keep_after_commit": 2,
                "prune_margin": 0.04,
                "decisive_gap": 0.03,
                "refine_gap": 0.0,
                "uncertainty_gate": 0.20,
                "max_children": 0,
                "max_refine_anchors": 0,
                "refine_time": 0.8,
                "refine_eval_time": 0.9,
                "tail_targets": (),
                "noise_policy": "inferred_noise",
                "strength": 0.2,
                "sde_scouts": 0,
                "sde_noise_scale": 0.0,
            }
        if beta <= 0.5:
            return {
                "target_nfe": target_nfe,
                "roots": 5,
                "scout_time": 0.3,
                "keep_after_scout": 3,
                "commit_count": 2,
                "commit_time": 0.7,
                "decisive_time": 0.8,
                "keep_after_commit": 2,
                "prune_margin": 0.035,
                "decisive_gap": 0.025,
                "refine_gap": 0.020,
                "uncertainty_gate": 0.18,
                "max_children": 2,
                "max_refine_anchors": 2,
                "refine_time": 0.6,
                "refine_eval_time": 0.8,
                "tail_targets": (1.0, 0.8),
                "noise_policy": "fresh_noise",
                "strength": 1.0,
                "sde_scouts": 1,
                "sde_noise_scale": 0.02,
            }
        if beta <= 0.75:
            return {
                "target_nfe": target_nfe,
                "roots": 6,
                "scout_time": 0.3,
                "keep_after_scout": 3,
                "commit_count": 3,
                "commit_time": 0.7,
                "decisive_time": 0.8,
                "keep_after_commit": 3,
                "prune_margin": 0.028,
                "decisive_gap": 0.020,
                "refine_gap": 0.018,
                "uncertainty_gate": 0.16,
                "max_children": 2,
                "max_refine_anchors": 2,
                "refine_time": 0.6,
                "refine_eval_time": 0.9,
                "tail_targets": (1.0, 0.9, 0.8),
                "noise_policy": "mixed_noise",
                "strength": 0.45,
                "sde_scouts": 2,
                "sde_noise_scale": 0.015,
            }
        return {
            "target_nfe": min(target_nfe, 64),
            "roots": 7,
            "scout_time": 0.3,
            "keep_after_scout": 4,
            "commit_count": 4,
            "commit_time": 0.7,
            "decisive_time": 0.8,
            "keep_after_commit": 4,
            "prune_margin": 0.024,
            "decisive_gap": 0.016,
            "refine_gap": 0.016,
            "uncertainty_gate": 0.15,
            "max_children": 3,
            "max_refine_anchors": 2,
            "refine_time": 0.6,
            "refine_eval_time": 0.9,
            "tail_targets": (1.0, 1.0, 0.9, 0.8),
            "noise_policy": "mixed_noise",
            "strength": 0.35,
            "sde_scouts": 2,
            "sde_noise_scale": 0.012,
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

    def _commit_stage(
        self,
        env: FlowTTSEnv,
        scout_survivors: list[int],
        schedule: dict[str, float | int | tuple[float, ...] | bool],
        initial_budget: int,
    ) -> list[int]:
        ranked = self._rank_previews(env, scout_survivors)
        if not ranked:
            return scout_survivors[:1]

        if self._should_go_deep(ranked, schedule):
            best_id = ranked[0].particle_id
            deep_cost = self._segment_cost(env, best_id, float(schedule["decisive_time"])) + 1
            if self._can_spend(env, initial_budget, schedule, deep_cost):
                self._forward_to(env, best_id, float(schedule["decisive_time"]), solver="euler")
                self._preview(env, best_id)
                self._prune_non_survivors(env, scout_survivors, [best_id])
                return [best_id]

        committed: list[int] = []
        keep_count = max(1, int(schedule["commit_count"]))
        for particle_id in self._rank_particle_ids_from_state(env, scout_survivors)[:keep_count]:
            commit_cost = self._segment_cost(env, particle_id, float(schedule["commit_time"])) + 1
            if not self._can_spend(env, initial_budget, schedule, commit_cost):
                break
            try:
                self._forward_to(env, particle_id, float(schedule["commit_time"]), solver="euler")
                self._preview(env, particle_id)
                committed.append(particle_id)
            except (BudgetExceededError, InvalidActionError):
                break

        if not committed:
            return scout_survivors[:1]

        ranked_ids = self._rank_particle_ids_from_state(env, committed)
        survivors = self._adaptive_keep(
            env,
            ranked_ids,
            max(1, int(schedule["keep_after_commit"])),
            float(schedule["prune_margin"]),
        )
        self._prune_non_survivors(env, scout_survivors, survivors)
        return survivors

    def _refine_stage(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | tuple[float, ...] | bool],
        initial_budget: int,
    ) -> list[int]:
        max_children = int(schedule["max_children"])
        max_refine_anchors = int(schedule["max_refine_anchors"])
        if max_children <= 0 or max_refine_anchors <= 0:
            return particle_ids

        ranked = self._rank_previews(env, particle_ids)
        if len(ranked) < 2:
            return particle_ids
        if not self._needs_repair(ranked, schedule):
            return particle_ids

        remaining_children = max_children
        new_ids: list[int] = []
        runner_gap = float(ranked[0].score or 0.0) - float(ranked[1].score or 0.0)

        for anchor_index, preview in enumerate(ranked[:max_refine_anchors]):
            if remaining_children <= 0:
                break
            gap = float(ranked[0].score or 0.0) - float(preview.score or 0.0)
            uncertainty = float(preview.uncertainty or 0.0)
            if anchor_index > 0 and gap > float(schedule["refine_gap"]) and uncertainty < float(schedule["uncertainty_gate"]):
                continue

            requested = 1
            if (
                anchor_index == 0
                and remaining_children >= 2
                and runner_gap <= float(schedule["refine_gap"])
                and uncertainty >= 0.5 * float(schedule["uncertainty_gate"])
            ):
                requested = 2

            child_cost = self._child_eval_cost(
                env,
                float(schedule["refine_time"]),
                float(schedule["refine_eval_time"]),
            )
            affordable = min(
                requested,
                remaining_children,
                self._max_affordable_children(env, initial_budget, schedule, child_cost),
            )
            if affordable <= 0:
                break

            try:
                children = env.backward(
                    preview.id,
                    target_time=float(schedule["refine_time"]),
                    noise_policy=str(schedule["noise_policy"]),
                    num_children=affordable,
                    strength=float(schedule["strength"]),
                )
            except (BudgetExceededError, InvalidActionError):
                break

            for child_id in children:
                try:
                    self._forward_to(
                        env,
                        child_id,
                        float(schedule["refine_eval_time"]),
                        solver="euler",
                    )
                    self._preview(env, child_id)
                    new_ids.append(child_id)
                except (BudgetExceededError, InvalidActionError):
                    return particle_ids + new_ids

            remaining_children -= affordable

        return particle_ids + new_ids

    def _tail_stage(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | tuple[float, ...] | bool],
        initial_budget: int,
    ) -> None:
        tail_targets = tuple(float(t) for t in schedule["tail_targets"])
        if not tail_targets:
            return

        for rank_index, target_time in enumerate(tail_targets):
            ranked_ids = self._rank_particle_ids_from_state(env, particle_ids)
            if rank_index >= len(ranked_ids):
                return
            particle_id = ranked_ids[rank_index]
            state = env.get_state()
            particle = state.particles.get(particle_id)
            if particle is None or particle.status != "active" or particle.time >= target_time:
                continue

            tail_cost = self._segment_cost(env, particle_id, target_time) + 1
            if not self._can_spend(env, initial_budget, schedule, tail_cost):
                continue

            try:
                self._forward_to(env, particle_id, target_time, solver="euler")
                self._preview(env, particle_id)
            except (BudgetExceededError, InvalidActionError):
                return

    def _fill_with_late_steps(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | tuple[float, ...] | bool],
        initial_budget: int,
    ) -> None:
        while True:
            if not self._can_spend(env, initial_budget, schedule, 2):
                return
            candidate_id, next_time = self._best_next_step(env, particle_ids)
            if candidate_id is None or next_time is None:
                return
            try:
                self._forward_to(env, candidate_id, next_time, solver="euler")
                self._preview(env, candidate_id)
            except (BudgetExceededError, InvalidActionError):
                return

    def _should_go_deep(
        self,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int | tuple[float, ...] | bool],
    ) -> bool:
        if len(ranked) < 2:
            return True
        best = ranked[0]
        runner_up = ranked[1]
        score_gap = float(best.score or 0.0) - float(runner_up.score or 0.0)
        best_uncertainty = float(best.uncertainty or 0.0)
        return score_gap >= float(schedule["decisive_gap"]) and best_uncertainty <= float(
            schedule["uncertainty_gate"]
        )

    def _needs_repair(
        self,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int | tuple[float, ...] | bool],
    ) -> bool:
        if len(ranked) < 2:
            return False
        best = ranked[0]
        runner_up = ranked[1]
        score_gap = float(best.score or 0.0) - float(runner_up.score or 0.0)
        uncertainty = max(float(best.uncertainty or 0.0), float(runner_up.uncertainty or 0.0))
        return score_gap <= float(schedule["refine_gap"]) or uncertainty >= float(schedule["uncertainty_gate"])

    def _adaptive_keep(
        self,
        env: FlowTTSEnv,
        ranked_ids: list[int],
        base_keep: int,
        prune_margin: float,
    ) -> list[int]:
        keep = max(1, min(base_keep, len(ranked_ids)))
        previews = self._rank_previews(env, ranked_ids)
        if not previews:
            return ranked_ids[:keep]
        threshold = float(previews[min(keep, len(previews)) - 1].score or 0.0)
        survivors: list[int] = []
        for particle_id in ranked_ids:
            preview = self._latest_preview_for_particle(env, particle_id)
            if preview is None or preview.score is None:
                continue
            if len(survivors) < keep or float(preview.score) >= threshold - prune_margin:
                survivors.append(particle_id)
        return survivors[: max(keep, len(survivors))]

    def _prune_non_survivors(self, env: FlowTTSEnv, all_ids: list[int], survivor_ids: list[int]) -> None:
        survivor_set = set(survivor_ids)
        state = env.get_state()
        to_prune = [
            particle_id
            for particle_id in all_ids
            if particle_id not in survivor_set
            and particle_id in state.particles
            and state.particles[particle_id].status == "active"
        ]
        if not to_prune:
            return
        try:
            env.prune(to_prune)
        except InvalidActionError:
            return

    def _forward_to(
        self,
        env: FlowTTSEnv,
        particle_id: int,
        target_time: float,
        solver: str,
        cfg: dict[str, float] | None = None,
    ) -> None:
        state = env.get_state()
        particle = state.particles[particle_id]
        for time_value in env.time_grid:
            if time_value > particle.time and time_value <= target_time + 1e-9:
                env.forward(particle_id, target_time=time_value, solver=solver, cfg=cfg)
                particle = env.get_state().particles[particle_id]
        if particle.time + 1e-9 < target_time <= 1.0 + 1e-9:
            env.forward(
                particle_id,
                target_time=min(1.0, float(target_time)),
                solver=solver,
                cfg=cfg,
            )

    def _preview(self, env: FlowTTSEnv, particle_id: int) -> PreviewRecord:
        return env.preview(particle_id, mode="clean_anchor", scorer="default")

    def _rank_particle_ids_from_state(self, env: FlowTTSEnv, particle_ids: list[int]) -> list[int]:
        previews = self._rank_previews(env, particle_ids)
        ranked = [preview.particle_id for preview in previews]
        missing = [particle_id for particle_id in particle_ids if particle_id not in ranked]
        return ranked + missing

    def _rank_previews(self, env: FlowTTSEnv, particle_ids: list[int]) -> list[PreviewRecord]:
        previews = [
            preview
            for particle_id in particle_ids
            for preview in [self._latest_preview_for_particle(env, particle_id)]
            if preview is not None and preview.score is not None
        ]
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

    def _latest_preview_for_particle(self, env: FlowTTSEnv, particle_id: int) -> PreviewRecord | None:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None or particle.last_preview_id is None:
            return None
        return state.previews.get(particle.last_preview_id)

    def _best_next_step(self, env: FlowTTSEnv, particle_ids: list[int]) -> tuple[int | None, float | None]:
        state = env.get_state()
        ranked_ids = self._rank_particle_ids_from_state(env, particle_ids)
        for particle_id in ranked_ids:
            particle = state.particles.get(particle_id)
            if particle is None or particle.status != "active":
                continue
            for time_value in env.time_grid:
                if time_value > particle.time + 1e-9:
                    return particle_id, float(time_value)
            if particle.time < 1.0 - 1e-9:
                return particle_id, 1.0
        return None, None

    def _segment_cost(self, env: FlowTTSEnv, particle_id: int, target_time: float) -> int:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None:
            return 0
        return sum(1 for time_value in env.time_grid if particle.time < time_value <= target_time + 1e-9)

    def _child_eval_cost(self, env: FlowTTSEnv, child_time: float, eval_time: float) -> int:
        forward_steps = sum(1 for time_value in env.time_grid if child_time < time_value <= eval_time + 1e-9)
        return forward_steps + 1

    def _max_affordable_children(
        self,
        env: FlowTTSEnv,
        initial_budget: int,
        schedule: dict[str, float | int | tuple[float, ...] | bool],
        child_cost: int,
    ) -> int:
        if child_cost <= 0:
            return 0
        remaining = int(schedule["target_nfe"]) - self._spent(env, initial_budget)
        return max(0, remaining // child_cost)

    def _can_spend(
        self,
        env: FlowTTSEnv,
        initial_budget: int,
        schedule: dict[str, float | int | tuple[float, ...] | bool],
        extra_cost: int,
    ) -> bool:
        if extra_cost < 0:
            return False
        spent = self._spent(env, initial_budget)
        return env.budget_left >= extra_cost and spent + extra_cost <= int(schedule["target_nfe"])

    def _spent(self, env: FlowTTSEnv, initial_budget: int) -> int:
        return max(0, int(initial_budget - env.budget_left))

    def _finish_one(self, env: FlowTTSEnv, particle_id: int | None) -> AnswerRecord:
        if particle_id is None:
            return self._safe_answer(env)
        try:
            self._forward_to(env, particle_id, 1.0, solver="euler")
            self._preview(env, particle_id)
            return env.answer(rule="best_preview_score")
        except (BudgetExceededError, InvalidActionError):
            return self._safe_answer(env)

    def _safe_answer(self, env: FlowTTSEnv) -> AnswerRecord:
        state = env.get_state()
        if state.previews:
            return env.answer(rule="best_preview_score")
        return env.answer(rule="latest_active")
