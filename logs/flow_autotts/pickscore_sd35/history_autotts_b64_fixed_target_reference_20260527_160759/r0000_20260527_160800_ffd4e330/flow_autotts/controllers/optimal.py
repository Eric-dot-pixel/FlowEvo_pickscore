"""Candidate controller for SD3.5 PickScore discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Beta-scheduled scout-confirm-refine controller."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        schedule = self._schedule(beta, env.budget)
        initial_budget = int(env.budget_left)

        try:
            scout_ids = env.spawn(int(schedule["roots"]))
        except InvalidActionError:
            return env.answer(rule="latest_active")

        active_ids = list(scout_ids)
        scout_previews: dict[int, PreviewRecord] = {}

        try:
            for particle_id in list(active_ids):
                if not self._can_spend(env, initial_budget, schedule, self._scout_cost(env, schedule)):
                    break
                preview = self._run_scout(env, particle_id, schedule)
                if preview is not None:
                    scout_previews[particle_id] = preview
        except (BudgetExceededError, InvalidActionError):
            return self._safe_answer(env)

        if not scout_previews:
            return self._finish_one(env, scout_ids[0] if scout_ids else None)

        ranked_ids = self._rank_particle_ids(env, scout_previews)
        survivors = ranked_ids[: int(schedule["keep_after_scout"])]
        self._prune_non_survivors(env, active_ids, survivors)
        active_ids = list(survivors)

        try:
            active_ids = self._advance_survivors(env, active_ids, schedule, initial_budget)
            self._late_confirm(env, active_ids, schedule, initial_budget)
            active_ids = self._selective_refine(env, active_ids, schedule, initial_budget)
            self._finalize_survivors(env, active_ids, schedule, initial_budget)
            self._opportunistic_confirm(env, active_ids, schedule, initial_budget)
        except BudgetExceededError:
            return self._safe_answer(env)

        return self._safe_answer(env)

    def _schedule(self, beta: float, budget: int) -> dict[str, float | int | bool | str]:
        if beta <= 0.0:
            target_nfe = min(int(budget), 10)
            return {
                "target_nfe": target_nfe,
                "roots": 2,
                "scout_time": 0.4,
                "commit_time": 1.0,
                "refine_time": 0.8,
                "keep_after_scout": 1,
                "keep_after_commit": 1,
                "base_children": 0,
                "max_refined_anchors": 0,
                "late_confirms": 0,
                "opportunistic_confirms": 0,
                "prune_margin": 0.08,
                "confirm_gap": 0.03,
                "refine_gap": 0.0,
                "uncertainty_gate": 0.22,
                "diversity_floor": 1,
                "noise_policy": "inferred_noise",
                "strength": 0.2,
                "solver": "euler",
                "scout_solver": "euler",
            }
        if beta <= 0.25:
            target_nfe = min(int(budget), 20)
            return {
                "target_nfe": target_nfe,
                "roots": 4,
                "scout_time": 0.4,
                "commit_time": 1.0,
                "refine_time": 0.8,
                "keep_after_scout": 2,
                "keep_after_commit": 1,
                "base_children": 0,
                "max_refined_anchors": 0,
                "late_confirms": 0,
                "opportunistic_confirms": 0,
                "prune_margin": 0.06,
                "confirm_gap": 0.025,
                "refine_gap": 0.0,
                "uncertainty_gate": 0.2,
                "diversity_floor": 1,
                "noise_policy": "inferred_noise",
                "strength": 0.2,
                "solver": "euler",
                "scout_solver": "euler",
            }
        if beta <= 0.5:
            target_nfe = min(int(budget), 36)
            return {
                "target_nfe": target_nfe,
                "roots": 5,
                "scout_time": 0.4,
                "commit_time": 0.8,
                "refine_time": 0.6,
                "keep_after_scout": 3,
                "keep_after_commit": 2,
                "base_children": 1,
                "max_refined_anchors": 1,
                "late_confirms": 1,
                "opportunistic_confirms": 1,
                "prune_margin": 0.045,
                "confirm_gap": 0.02,
                "refine_gap": 0.03,
                "uncertainty_gate": 0.18,
                "diversity_floor": 1,
                "noise_policy": "fresh_noise",
                "strength": 1.0,
                "solver": "euler",
                "scout_solver": "euler",
            }
        if beta <= 0.75:
            target_nfe = min(int(budget), 48)
            return {
                "target_nfe": target_nfe,
                "roots": 6,
                "scout_time": 0.4,
                "commit_time": 0.8,
                "refine_time": 0.5,
                "keep_after_scout": 3,
                "keep_after_commit": 2,
                "base_children": 1,
                "max_refined_anchors": 2,
                "late_confirms": 2,
                "opportunistic_confirms": 2,
                "prune_margin": 0.035,
                "confirm_gap": 0.018,
                "refine_gap": 0.028,
                "uncertainty_gate": 0.15,
                "diversity_floor": 2,
                "noise_policy": "mixed_noise",
                "strength": 0.45,
                "solver": "euler",
                "scout_solver": "sde",
            }
        target_nfe = min(int(budget), 64)
        return {
            "target_nfe": target_nfe,
            "roots": 7,
            "scout_time": 0.4,
            "commit_time": 0.7,
            "refine_time": 0.5,
            "keep_after_scout": 4,
            "keep_after_commit": 3,
            "base_children": 2,
            "max_refined_anchors": 2,
            "late_confirms": 3,
            "opportunistic_confirms": 2,
            "prune_margin": 0.028,
            "confirm_gap": 0.015,
            "refine_gap": 0.025,
            "uncertainty_gate": 0.14,
            "diversity_floor": 2,
            "noise_policy": "mixed_noise",
            "strength": 0.35,
            "solver": "euler",
            "scout_solver": "sde",
        }

    def _run_scout(
        self,
        env: FlowTTSEnv,
        particle_id: int,
        schedule: dict[str, float | int | bool | str],
    ) -> PreviewRecord | None:
        self._forward_to(
            env,
            particle_id,
            float(schedule["scout_time"]),
            solver=str(schedule["scout_solver"]),
        )
        return self._preview(env, particle_id)

    def _advance_survivors(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | bool | str],
        initial_budget: int,
    ) -> list[int]:
        committed: list[int] = []
        ranked = self._rank_particle_ids_from_state(env, particle_ids)
        for particle_id in ranked:
            if not self._can_spend(env, initial_budget, schedule, self._commit_cost(env, particle_id, schedule)):
                break
            try:
                self._forward_to(
                    env,
                    particle_id,
                    float(schedule["commit_time"]),
                    solver=str(schedule["solver"]),
                )
                self._preview(env, particle_id)
                committed.append(particle_id)
            except (BudgetExceededError, InvalidActionError):
                break

        if committed:
            keep = max(int(schedule["keep_after_commit"]), int(schedule["diversity_floor"]))
            ranked_commit = self._rank_particle_ids_from_state(env, committed)
            survivors = ranked_commit[:keep]
            self._prune_non_survivors(env, particle_ids, survivors)
            return survivors
        return particle_ids[: max(1, int(schedule["diversity_floor"]))]

    def _late_confirm(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | bool | str],
        initial_budget: int,
    ) -> None:
        if len(particle_ids) < 2 or int(schedule["late_confirms"]) <= 0:
            return

        ranked = self._rank_previews(env, particle_ids)
        if len(ranked) < 2:
            return
        best = ranked[0]
        runner_up = ranked[1]
        score_gap = float(best.score) - float(runner_up.score)
        uncertainty = max(float(best.uncertainty or 0.0), float(runner_up.uncertainty or 0.0))
        need_confirmation = (
            score_gap <= float(schedule["confirm_gap"]) or uncertainty >= float(schedule["uncertainty_gate"])
        )
        if not need_confirmation:
            return

        confirms = min(int(schedule["late_confirms"]), len(particle_ids))
        for preview in ranked[:confirms]:
            if not self._can_spend(env, initial_budget, schedule, self._preview_cost()):
                return
            try:
                self._preview(env, preview.particle_id)
            except (BudgetExceededError, InvalidActionError):
                return

    def _selective_refine(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | bool | str],
        initial_budget: int,
    ) -> list[int]:
        if int(schedule["base_children"]) <= 0 or int(schedule["max_refined_anchors"]) <= 0:
            return particle_ids

        ranked = self._rank_previews(env, particle_ids)
        if not ranked:
            return particle_ids

        best_score = float(ranked[0].score)
        refined_ids: list[int] = list(particle_ids)
        refined_anchors = 0

        for preview in ranked:
            if refined_anchors >= int(schedule["max_refined_anchors"]):
                break
            score_gap = best_score - float(preview.score)
            uncertainty = float(preview.uncertainty or 0.0)
            if score_gap > float(schedule["refine_gap"]) and uncertainty < float(schedule["uncertainty_gate"]):
                continue

            extra_child = 1 if uncertainty >= float(schedule["uncertainty_gate"]) * 0.9 else 0
            num_children = min(2, int(schedule["base_children"]) + extra_child)
            projected_cost = num_children * self._child_cost(env, float(schedule["refine_time"]))
            if not self._can_spend(env, initial_budget, schedule, projected_cost):
                break
            try:
                child_ids = env.backward(
                    preview.id,
                    target_time=float(schedule["refine_time"]),
                    noise_policy=str(schedule["noise_policy"]),
                    num_children=num_children,
                    strength=float(schedule["strength"]),
                )
            except (BudgetExceededError, InvalidActionError):
                break

            viable_children: list[int] = []
            for child_id in child_ids:
                if not self._can_spend(
                    env,
                    initial_budget,
                    schedule,
                    self._finish_cost(env, child_id) + self._preview_cost(),
                ):
                    break
                try:
                    self._forward_to(env, child_id, 1.0, solver=str(schedule["solver"]))
                    self._preview(env, child_id)
                    viable_children.append(child_id)
                except (BudgetExceededError, InvalidActionError):
                    break
            refined_ids.extend(viable_children)
            refined_anchors += 1

        ranked_all = self._rank_particle_ids_from_state(env, refined_ids)
        keep = max(int(schedule["keep_after_commit"]), int(schedule["diversity_floor"]))
        survivors = ranked_all[: max(1, keep)]
        self._prune_non_survivors(env, refined_ids, survivors)
        return survivors

    def _finalize_survivors(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | bool | str],
        initial_budget: int,
    ) -> None:
        ranked = self._rank_particle_ids_from_state(env, particle_ids)
        for particle_id in ranked:
            if not self._can_spend(env, initial_budget, schedule, self._finish_and_preview_cost(env, particle_id)):
                continue
            try:
                self._forward_to(env, particle_id, 1.0, solver=str(schedule["solver"]))
                self._preview(env, particle_id)
            except (BudgetExceededError, InvalidActionError):
                return

    def _opportunistic_confirm(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | bool | str],
        initial_budget: int,
    ) -> None:
        previews = self._rank_previews(env, particle_ids)
        if not previews:
            return

        target_nfe = int(schedule["target_nfe"])
        confirm_budget = int(schedule["opportunistic_confirms"])
        if env.budget_left <= 0 or confirm_budget <= 0:
            return

        if self._spent(env, initial_budget) >= target_nfe:
            return

        ranked_ids = [preview.particle_id for preview in previews]
        for particle_id in ranked_ids[:confirm_budget]:
            if not self._can_spend(env, initial_budget, schedule, self._preview_cost()):
                return
            try:
                self._preview(env, particle_id)
            except (BudgetExceededError, InvalidActionError):
                return

    def _finish_one(self, env: FlowTTSEnv, particle_id: int | None) -> AnswerRecord:
        if particle_id is not None:
            try:
                self._forward_to(env, particle_id, 1.0, solver="euler")
                self._preview(env, particle_id)
                return env.answer(rule="best_preview_score")
            except (BudgetExceededError, InvalidActionError):
                return self._safe_answer(env)
        return self._safe_answer(env)

    def _safe_answer(self, env: FlowTTSEnv) -> AnswerRecord:
        state = env.get_state()
        if state.previews:
            return env.answer(rule="best_preview_score")
        return env.answer(rule="latest_active")

    def _forward_to(self, env: FlowTTSEnv, particle_id: int, target_time: float, solver: str) -> None:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None or particle.status == "pruned":
            raise InvalidActionError("particle unavailable for forward")
        if particle.time >= target_time:
            return
        env.forward(particle_id, target_time=target_time, solver=solver, cfg=self._solver_cfg(solver))

    def _preview(self, env: FlowTTSEnv, particle_id: int) -> PreviewRecord:
        return env.preview(particle_id, mode="clean_anchor", scorer="default")

    def _rank_particle_ids(
        self,
        env: FlowTTSEnv,
        previews_by_particle: dict[int, PreviewRecord],
    ) -> list[int]:
        state = env.get_state()
        previews = [
            preview
            for particle_id, preview in previews_by_particle.items()
            if particle_id in state.particles and state.particles[particle_id].status != "pruned"
        ]
        ranked = self._sort_previews(previews)
        return [preview.particle_id for preview in ranked]

    def _rank_particle_ids_from_state(self, env: FlowTTSEnv, particle_ids: list[int]) -> list[int]:
        previews = self._rank_previews(env, particle_ids)
        ranked_ids = [preview.particle_id for preview in previews]
        seen = set(ranked_ids)
        state = env.get_state()
        fallback = [
            particle_id
            for particle_id in particle_ids
            if particle_id in state.particles and state.particles[particle_id].status != "pruned" and particle_id not in seen
        ]
        fallback.sort(
            key=lambda particle_id: (
                float(state.particles[particle_id].time),
                -float(state.particles[particle_id].num_children),
                -int(particle_id),
            ),
            reverse=True,
        )
        return ranked_ids + fallback

    def _rank_previews(self, env: FlowTTSEnv, particle_ids: list[int]) -> list[PreviewRecord]:
        state = env.get_state()
        keep = set(particle_ids)
        previews = [
            preview
            for preview in state.previews.values()
            if preview.particle_id in keep and state.particles[preview.particle_id].status != "pruned"
        ]
        latest: dict[int, PreviewRecord] = {}
        for preview in previews:
            current = latest.get(preview.particle_id)
            if current is None or preview.id > current.id:
                latest[preview.particle_id] = preview
        return self._sort_previews(list(latest.values()))

    def _sort_previews(self, previews: list[PreviewRecord]) -> list[PreviewRecord]:
        return sorted(
            [preview for preview in previews if preview.score is not None],
            key=lambda preview: (
                float(preview.score),
                -float(preview.uncertainty or 0.0),
                -float(preview.drift or 0.0),
                float(preview.time),
            ),
            reverse=True,
        )

    def _prune_non_survivors(
        self,
        env: FlowTTSEnv,
        candidate_ids: list[int],
        survivor_ids: list[int],
    ) -> None:
        state = env.get_state()
        survivor_set = set(survivor_ids)
        prune_ids = [
            particle_id
            for particle_id in candidate_ids
            if particle_id in state.particles and state.particles[particle_id].status == "active" and particle_id not in survivor_set
        ]
        if not prune_ids:
            return
        try:
            env.prune(prune_ids)
        except InvalidActionError:
            return

    def _can_spend(
        self,
        env: FlowTTSEnv,
        initial_budget: int,
        schedule: dict[str, float | int | bool | str],
        cost: int,
    ) -> bool:
        if cost <= 0:
            return True
        target_left = int(schedule["target_nfe"]) - self._spent(env, initial_budget)
        return env.budget_left >= cost and target_left >= cost

    def _spent(self, env: FlowTTSEnv, initial_budget: int) -> int:
        return max(0, int(initial_budget - env.budget_left))

    def _scout_cost(self, env: FlowTTSEnv, schedule: dict[str, float | int | bool | str]) -> int:
        scout_steps = self._step_count(
            env_time_grid=env.time_grid,
            start_time=0.0,
            target_time=float(schedule["scout_time"]),
        )
        return scout_steps + self._preview_cost()

    def _commit_cost(
        self,
        env: FlowTTSEnv,
        particle_id: int,
        schedule: dict[str, float | int | bool | str],
    ) -> int:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None:
            return 0
        steps = self._step_count(
            env_time_grid=env.time_grid,
            start_time=float(particle.time),
            target_time=float(schedule["commit_time"]),
        )
        return steps + self._preview_cost()

    def _finish_cost(self, env: FlowTTSEnv, particle_id: int) -> int:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None:
            return 0
        return self._step_count(
            env_time_grid=env.time_grid,
            start_time=float(particle.time),
            target_time=1.0,
        )

    def _finish_and_preview_cost(self, env: FlowTTSEnv, particle_id: int) -> int:
        return self._finish_cost(env, particle_id) + self._preview_cost()

    def _child_cost(self, env: FlowTTSEnv, target_time: float) -> int:
        return self._step_count(
            env_time_grid=env.time_grid,
            start_time=target_time,
            target_time=1.0,
        ) + self._preview_cost()

    def _preview_cost(self) -> int:
        return 1

    def _step_count(
        self,
        env_time_grid: tuple[float, ...] | None,
        start_time: float,
        target_time: float,
    ) -> int:
        grid = tuple(float(t) for t in env_time_grid) if env_time_grid is not None else (0.0, 1.0)
        start = float(start_time)
        target = float(target_time)
        if target <= start:
            return 0
        return sum(1 for time in grid if start < float(time) <= target)

    def _solver_cfg(self, solver: str) -> dict[str, float | str] | None:
        if solver != "sde":
            return None
        return {"noise_scale": 0.01, "sigma_max": 1.25, "min_time": 0.02}
