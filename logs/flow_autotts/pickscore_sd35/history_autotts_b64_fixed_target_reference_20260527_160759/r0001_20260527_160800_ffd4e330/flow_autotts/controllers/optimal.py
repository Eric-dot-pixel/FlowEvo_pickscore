"""Candidate controller for SD3.5 PickScore discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Beta-scheduled scout, confirm, and selective local-refine controller."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        schedule = self._schedule(beta, int(env.budget))
        initial_budget = int(env.budget_left)

        try:
            root_ids = env.spawn(int(schedule["roots"]))
        except InvalidActionError:
            return self._safe_answer(env)

        scout_ids: list[int] = []
        for particle_id in root_ids:
            if not self._can_spend(env, initial_budget, schedule, self._scout_cost(env, schedule)):
                break
            try:
                self._forward_to(
                    env,
                    particle_id,
                    float(schedule["scout_time"]),
                    solver=str(schedule["scout_solver"]),
                )
                self._preview(env, particle_id)
                scout_ids.append(particle_id)
            except (BudgetExceededError, InvalidActionError):
                break

        if not scout_ids:
            return self._finish_one(env, root_ids[0] if root_ids else None)

        ranked_scouts = self._rank_particle_ids_from_state(env, scout_ids)
        survivors = ranked_scouts[: max(1, int(schedule["keep_after_scout"]))]
        self._prune_non_survivors(env, scout_ids, survivors)

        try:
            committed = self._commit_stage(env, survivors, schedule, initial_budget)
            self._confirm_ambiguous(env, committed, schedule, initial_budget)
            active = self._refine_stage(env, committed, schedule, initial_budget)
            self._final_confirm(env, active, schedule, initial_budget)
            self._fill_to_target(env, active, schedule, initial_budget)
        except BudgetExceededError:
            return self._safe_answer(env)

        return self._safe_answer(env)

    def _schedule(self, beta: float, budget: int) -> dict[str, float | int | str]:
        target_nfe = self._target_nfe(budget, beta)
        if beta <= 0.0:
            return {
                "target_nfe": target_nfe,
                "roots": 2,
                "scout_time": 0.4,
                "commit_time": 0.8,
                "refine_time": 0.8,
                "parent_finish_time": 1.0,
                "keep_after_scout": 1,
                "keep_after_commit": 1,
                "max_children": 0,
                "max_refine_anchors": 0,
                "late_confirms": 0,
                "target_confirms": 0,
                "prune_margin": 0.10,
                "confirm_gap": 0.030,
                "refine_gap": 0.000,
                "uncertainty_gate": 0.70,
                "noise_policy": "inferred_noise",
                "strength": 0.20,
                "scout_solver": "euler",
            }
        if beta <= 0.25:
            return {
                "target_nfe": target_nfe,
                "roots": 4,
                "scout_time": 0.4,
                "commit_time": 0.8,
                "refine_time": 0.8,
                "parent_finish_time": 1.0,
                "keep_after_scout": 2,
                "keep_after_commit": 2,
                "max_children": 1,
                "max_refine_anchors": 1,
                "late_confirms": 1,
                "target_confirms": 0,
                "prune_margin": 0.050,
                "confirm_gap": 0.022,
                "refine_gap": 0.015,
                "uncertainty_gate": 0.45,
                "noise_policy": "inferred_noise",
                "strength": 0.22,
                "scout_solver": "euler",
            }
        if beta <= 0.5:
            return {
                "target_nfe": target_nfe,
                "roots": 4,
                "scout_time": 0.4,
                "commit_time": 0.8,
                "refine_time": 0.8,
                "parent_finish_time": 1.0,
                "keep_after_scout": 2,
                "keep_after_commit": 2,
                "max_children": 2,
                "max_refine_anchors": 1,
                "late_confirms": 1,
                "target_confirms": 1,
                "prune_margin": 0.040,
                "confirm_gap": 0.018,
                "refine_gap": 0.018,
                "uncertainty_gate": 0.38,
                "noise_policy": "fresh_noise",
                "strength": 1.00,
                "scout_solver": "euler",
            }
        if beta <= 0.75:
            return {
                "target_nfe": target_nfe,
                "roots": 5,
                "scout_time": 0.4,
                "commit_time": 0.6,
                "refine_time": 0.6,
                "parent_finish_time": 1.0,
                "keep_after_scout": 3,
                "keep_after_commit": 2,
                "max_children": 3,
                "max_refine_anchors": 2,
                "late_confirms": 2,
                "target_confirms": 1,
                "prune_margin": 0.030,
                "confirm_gap": 0.015,
                "refine_gap": 0.020,
                "uncertainty_gate": 0.34,
                "noise_policy": "mixed_noise",
                "strength": 0.40,
                "scout_solver": "sde",
            }
        return {
            "target_nfe": min(target_nfe, 64),
            "roots": 6,
            "scout_time": 0.3,
            "commit_time": 0.6,
            "refine_time": 0.6,
            "parent_finish_time": 1.0,
            "keep_after_scout": 3,
            "keep_after_commit": 3,
            "max_children": 4,
            "max_refine_anchors": 2,
            "late_confirms": 2,
            "target_confirms": 2,
            "prune_margin": 0.024,
            "confirm_gap": 0.013,
            "refine_gap": 0.018,
            "uncertainty_gate": 0.30,
            "noise_policy": "mixed_noise",
            "strength": 0.35,
            "scout_solver": "sde",
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
        particle_ids: list[int],
        schedule: dict[str, float | int | str],
        initial_budget: int,
    ) -> list[int]:
        committed: list[int] = []
        for particle_id in self._rank_particle_ids_from_state(env, particle_ids):
            if not self._can_spend(env, initial_budget, schedule, self._commit_cost(env, particle_id, schedule)):
                break
            try:
                self._forward_to(env, particle_id, float(schedule["commit_time"]), solver="euler")
                self._preview(env, particle_id)
                committed.append(particle_id)
            except (BudgetExceededError, InvalidActionError):
                break

        if not committed:
            return particle_ids[:1]

        ranked = self._rank_particle_ids_from_state(env, committed)
        keep = max(1, int(schedule["keep_after_commit"]))
        survivors = self._adaptive_keep(env, ranked, keep, float(schedule["prune_margin"]))
        self._prune_non_survivors(env, particle_ids, survivors)
        return survivors

    def _confirm_ambiguous(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | str],
        initial_budget: int,
    ) -> None:
        budgeted = int(schedule["late_confirms"])
        if budgeted <= 0 or len(particle_ids) < 2:
            return
        previews = self._rank_previews(env, particle_ids)
        if len(previews) < 2:
            return
        if not self._needs_confirmation(previews, schedule):
            return
        for preview in previews[:budgeted]:
            if not self._can_spend(env, initial_budget, schedule, self._preview_cost()):
                return
            try:
                self._preview(env, preview.particle_id)
            except (BudgetExceededError, InvalidActionError):
                return

    def _refine_stage(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | str],
        initial_budget: int,
    ) -> list[int]:
        max_children = int(schedule["max_children"])
        max_refine_anchors = int(schedule["max_refine_anchors"])
        if max_children <= 0 or max_refine_anchors <= 0:
            return particle_ids

        ranked = self._rank_previews(env, particle_ids)
        if not ranked:
            return particle_ids

        survivors = list(particle_ids)
        remaining_children = max_children
        refined_anchors = 0
        best_score = float(ranked[0].score or 0.0)

        for preview in ranked:
            if remaining_children <= 0 or refined_anchors >= max_refine_anchors:
                break
            score_gap = best_score - float(preview.score or 0.0)
            uncertainty = float(preview.uncertainty or 0.0)
            if score_gap > float(schedule["refine_gap"]) and uncertainty < float(schedule["uncertainty_gate"]):
                continue

            child_count = self._children_for_anchor(
                rank_index=refined_anchors,
                score_gap=score_gap,
                uncertainty=uncertainty,
                schedule=schedule,
                remaining_children=remaining_children,
            )
            if child_count <= 0:
                continue

            projected = child_count * self._child_cost(env, float(schedule["refine_time"]))
            if not self._can_spend(env, initial_budget, schedule, projected):
                break

            try:
                child_ids = env.backward(
                    preview.id,
                    target_time=float(schedule["refine_time"]),
                    noise_policy=str(schedule["noise_policy"]),
                    num_children=child_count,
                    strength=float(schedule["strength"]),
                )
            except (BudgetExceededError, InvalidActionError):
                break

            kept_children: list[int] = []
            for child_id in child_ids:
                finish_cost = self._finish_cost(env, child_id) + self._preview_cost()
                if not self._can_spend(env, initial_budget, schedule, finish_cost):
                    break
                try:
                    self._forward_to(env, child_id, 1.0, solver="euler")
                    self._preview(env, child_id)
                    kept_children.append(child_id)
                except (BudgetExceededError, InvalidActionError):
                    break

            survivors.extend(kept_children)
            remaining_children -= len(kept_children)
            refined_anchors += 1

        ranked_all = self._rank_particle_ids_from_state(env, survivors)
        keep = max(1, int(schedule["keep_after_commit"]))
        diversified = self._adaptive_keep(env, ranked_all, keep, float(schedule["prune_margin"]))
        self._prune_non_survivors(env, survivors, diversified)
        return diversified

    def _children_for_anchor(
        self,
        rank_index: int,
        score_gap: float,
        uncertainty: float,
        schedule: dict[str, float | int | str],
        remaining_children: int,
    ) -> int:
        base = 1
        if rank_index == 0 and (uncertainty >= float(schedule["uncertainty_gate"]) or score_gap <= float(schedule["confirm_gap"])):
            base += 1
        if uncertainty >= float(schedule["uncertainty_gate"]) * 1.15:
            base += 1
        if rank_index > 0 and score_gap > float(schedule["confirm_gap"]) * 1.5:
            base -= 1
        return max(0, min(remaining_children, base))

    def _final_confirm(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | str],
        initial_budget: int,
    ) -> None:
        confirms = int(schedule["target_confirms"])
        if confirms <= 0:
            return
        ranked = self._rank_previews(env, particle_ids)
        if not ranked:
            return
        for preview in ranked[:confirms]:
            if not self._can_spend(env, initial_budget, schedule, self._preview_cost()):
                return
            try:
                self._preview(env, preview.particle_id)
            except (BudgetExceededError, InvalidActionError):
                return

    def _fill_to_target(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | str],
        initial_budget: int,
    ) -> None:
        ranked = self._rank_particle_ids_from_state(env, particle_ids)
        if not ranked:
            return

        parent_finish_time = float(schedule["parent_finish_time"])
        for particle_id in ranked:
            finish_cost = self._step_cost(env, particle_id, parent_finish_time)
            if finish_cost <= 0:
                continue
            total_cost = finish_cost + self._preview_cost()
            if not self._can_spend(env, initial_budget, schedule, total_cost):
                continue
            try:
                self._forward_to(env, particle_id, parent_finish_time, solver="euler")
                self._preview(env, particle_id)
            except (BudgetExceededError, InvalidActionError):
                return

        ranked_previews = self._rank_previews(env, ranked)
        if not ranked_previews:
            return

        while self._can_spend(env, initial_budget, schedule, self._preview_cost()):
            if not self._needs_more_target_spend(env, initial_budget, schedule):
                return
            refreshed = False
            for preview in ranked_previews[:2]:
                try:
                    self._preview(env, preview.particle_id)
                    refreshed = True
                except (BudgetExceededError, InvalidActionError):
                    continue
                if not self._needs_more_target_spend(env, initial_budget, schedule):
                    return
            if not refreshed:
                return
            ranked_previews = self._rank_previews(env, ranked)

    def _adaptive_keep(
        self,
        env: FlowTTSEnv,
        ranked_ids: list[int],
        base_keep: int,
        prune_margin: float,
    ) -> list[int]:
        previews = self._rank_previews(env, ranked_ids)
        if not previews:
            return ranked_ids[: max(1, base_keep)]
        cutoff = float(previews[0].score or 0.0) - float(prune_margin)
        kept = [
            preview.particle_id
            for preview in previews
            if float(preview.score or 0.0) >= cutoff
        ]
        return kept[: max(base_keep, len(kept))]

    def _needs_confirmation(
        self,
        previews: list[PreviewRecord],
        schedule: dict[str, float | int | str],
    ) -> bool:
        if len(previews) < 2:
            return False
        best = previews[0]
        runner_up = previews[1]
        gap = float(best.score or 0.0) - float(runner_up.score or 0.0)
        uncertainty = max(float(best.uncertainty or 0.0), float(runner_up.uncertainty or 0.0))
        return gap <= float(schedule["confirm_gap"]) or uncertainty >= float(schedule["uncertainty_gate"])

    def _needs_more_target_spend(
        self,
        env: FlowTTSEnv,
        initial_budget: int,
        schedule: dict[str, float | int | str],
    ) -> bool:
        return self._spent(env, initial_budget) + self._preview_cost() <= int(schedule["target_nfe"])

    def _finish_one(self, env: FlowTTSEnv, particle_id: int | None) -> AnswerRecord:
        if particle_id is None:
            return self._safe_answer(env)
        try:
            self._forward_to(env, particle_id, 1.0, solver="euler")
            self._preview(env, particle_id)
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

    def _rank_particle_ids_from_state(self, env: FlowTTSEnv, particle_ids: list[int]) -> list[int]:
        previews = self._rank_previews(env, particle_ids)
        ranked = [preview.particle_id for preview in previews]
        seen = set(ranked)
        state = env.get_state()
        fallback = [
            particle_id
            for particle_id in particle_ids
            if particle_id in state.particles
            and state.particles[particle_id].status != "pruned"
            and particle_id not in seen
        ]
        fallback.sort(
            key=lambda particle_id: (
                float(state.particles[particle_id].time),
                -float(state.particles[particle_id].num_children),
                -int(particle_id),
            ),
            reverse=True,
        )
        return ranked + fallback

    def _rank_previews(self, env: FlowTTSEnv, particle_ids: list[int]) -> list[PreviewRecord]:
        state = env.get_state()
        keep = set(particle_ids)
        latest: dict[int, PreviewRecord] = {}
        for preview in state.previews.values():
            if preview.particle_id not in keep:
                continue
            particle = state.particles.get(preview.particle_id)
            if particle is None or particle.status == "pruned":
                continue
            current = latest.get(preview.particle_id)
            if current is None or preview.id > current.id:
                latest[preview.particle_id] = preview
        return sorted(
            [preview for preview in latest.values() if preview.score is not None],
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
            if particle_id in state.particles
            and state.particles[particle_id].status == "active"
            and particle_id not in survivor_set
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
        schedule: dict[str, float | int | str],
        cost: int,
    ) -> bool:
        if cost <= 0:
            return True
        target_left = int(schedule["target_nfe"]) - self._spent(env, initial_budget)
        return env.budget_left >= cost and target_left >= cost

    def _spent(self, env: FlowTTSEnv, initial_budget: int) -> int:
        return max(0, int(initial_budget - env.budget_left))

    def _scout_cost(self, env: FlowTTSEnv, schedule: dict[str, float | int | str]) -> int:
        return self._step_count(env.time_grid, 0.0, float(schedule["scout_time"])) + self._preview_cost()

    def _commit_cost(
        self,
        env: FlowTTSEnv,
        particle_id: int,
        schedule: dict[str, float | int | str],
    ) -> int:
        return self._step_cost(env, particle_id, float(schedule["commit_time"])) + self._preview_cost()

    def _finish_cost(self, env: FlowTTSEnv, particle_id: int) -> int:
        return self._step_cost(env, particle_id, 1.0)

    def _step_cost(self, env: FlowTTSEnv, particle_id: int, target_time: float) -> int:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None or particle.status == "pruned":
            return 0
        return self._step_count(env.time_grid, float(particle.time), float(target_time))

    def _child_cost(self, env: FlowTTSEnv, target_time: float) -> int:
        return self._step_count(env.time_grid, float(target_time), 1.0) + self._preview_cost()

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
        return {"noise_scale": 0.008, "sigma_max": 1.25, "min_time": 0.02}
