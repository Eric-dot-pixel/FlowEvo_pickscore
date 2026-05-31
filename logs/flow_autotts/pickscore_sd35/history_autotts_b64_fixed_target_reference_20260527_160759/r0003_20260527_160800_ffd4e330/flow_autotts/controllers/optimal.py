"""Candidate controller for SD3.5 PickScore discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Beta-scheduled rival-preserving controller with selective late repair."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        initial_budget = int(env.budget_left)
        schedule = self._schedule(beta, int(env.budget))

        try:
            root_ids = env.spawn(int(schedule["roots"]))
        except InvalidActionError:
            return self._safe_answer(env)

        active_ids = list(root_ids)
        scout_ids: list[int] = []
        sde_scouts = int(schedule["sde_scouts"])

        for index, particle_id in enumerate(root_ids):
            scout_cost = self._segment_cost(env, particle_id, float(schedule["scout_time"])) + 1
            if not self._can_spend(env, initial_budget, schedule, scout_cost):
                break
            solver = "sde" if index < sde_scouts else "euler"
            try:
                self._forward_to(
                    env,
                    particle_id,
                    float(schedule["scout_time"]),
                    solver=solver,
                    cfg=self._sde_cfg(schedule) if solver == "sde" else None,
                )
                self._preview(env, particle_id)
                scout_ids.append(particle_id)
            except (BudgetExceededError, InvalidActionError):
                break

        if not scout_ids:
            return self._finish_one(env, active_ids[0] if active_ids else None)

        scout_survivors = self._adaptive_keep(
            env,
            scout_ids,
            int(schedule["keep_after_scout"]),
            float(schedule["scout_margin"]),
        )
        self._prune_non_survivors(env, active_ids, scout_survivors)
        active_ids = list(scout_survivors)

        try:
            active_ids = self._commit_stage(env, active_ids, schedule, initial_budget)
            self._late_confirm(env, active_ids, schedule, initial_budget)
            active_ids = self._selective_repair(env, active_ids, schedule, initial_budget)
            self._tail_stage(env, active_ids, schedule, initial_budget)
            self._fill_target(env, active_ids, schedule, initial_budget)
        except BudgetExceededError:
            return self._safe_answer(env)

        return self._safe_answer(env)

    def _schedule(self, beta: float, budget: int) -> dict[str, float | int | str]:
        if beta <= 0.0:
            return {
                "target_nfe": min(int(budget), 10),
                "roots": 2,
                "sde_scouts": 0,
                "sde_noise_scale": 0.0,
                "scout_time": 0.2,
                "commit_time": 0.5,
                "repair_time": 0.0,
                "repair_eval_time": 0.0,
                "keep_after_scout": 1,
                "keep_after_commit": 1,
                "scout_margin": 0.05,
                "commit_margin": 0.04,
                "confirm_gap": 0.018,
                "repair_gap": 0.0,
                "uncertainty_gate": 0.22,
                "max_confirmations": 1,
                "max_repair_children": 0,
                "noise_policy": "fresh_noise",
                "strength": 1.0,
                "tail_top_time": 1.0,
                "tail_rival_time": 0.0,
                "stop_spend_floor": 10,
            }
        if beta <= 0.25:
            return {
                "target_nfe": min(int(budget), 20),
                "roots": 4,
                "sde_scouts": 0,
                "sde_noise_scale": 0.0,
                "scout_time": 0.2,
                "commit_time": 0.5,
                "repair_time": 0.0,
                "repair_eval_time": 0.0,
                "keep_after_scout": 2,
                "keep_after_commit": 1,
                "scout_margin": 0.045,
                "commit_margin": 0.035,
                "confirm_gap": 0.020,
                "repair_gap": 0.0,
                "uncertainty_gate": 0.20,
                "max_confirmations": 2,
                "max_repair_children": 0,
                "noise_policy": "fresh_noise",
                "strength": 1.0,
                "tail_top_time": 1.0,
                "tail_rival_time": 0.0,
                "stop_spend_floor": 18,
            }
        if beta <= 0.5:
            return {
                "target_nfe": min(int(budget), 36),
                "roots": 5,
                "sde_scouts": 0,
                "sde_noise_scale": 0.0,
                "scout_time": 0.3,
                "commit_time": 0.7,
                "repair_time": 0.6,
                "repair_eval_time": 0.9,
                "keep_after_scout": 3,
                "keep_after_commit": 2,
                "scout_margin": 0.038,
                "commit_margin": 0.030,
                "confirm_gap": 0.020,
                "repair_gap": 0.014,
                "uncertainty_gate": 0.18,
                "max_confirmations": 2,
                "max_repair_children": 1,
                "noise_policy": "fresh_noise",
                "strength": 1.0,
                "tail_top_time": 1.0,
                "tail_rival_time": 0.9,
                "stop_spend_floor": 32,
            }
        if beta <= 0.75:
            return {
                "target_nfe": min(int(budget), 48),
                "roots": 5,
                "sde_scouts": 1,
                "sde_noise_scale": 0.010,
                "scout_time": 0.3,
                "commit_time": 0.7,
                "repair_time": 0.6,
                "repair_eval_time": 0.9,
                "keep_after_scout": 3,
                "keep_after_commit": 3,
                "scout_margin": 0.032,
                "commit_margin": 0.026,
                "confirm_gap": 0.018,
                "repair_gap": 0.012,
                "uncertainty_gate": 0.17,
                "max_confirmations": 2,
                "max_repair_children": 1,
                "noise_policy": "fresh_noise",
                "strength": 0.85,
                "tail_top_time": 1.0,
                "tail_rival_time": 1.0,
                "stop_spend_floor": 44,
            }
        return {
            "target_nfe": min(int(budget), 64),
            "roots": 6,
            "sde_scouts": 1,
            "sde_noise_scale": 0.008,
            "scout_time": 0.3,
            "commit_time": 0.7,
            "repair_time": 0.6,
            "repair_eval_time": 0.9,
            "keep_after_scout": 4,
            "keep_after_commit": 4,
            "scout_margin": 0.028,
            "commit_margin": 0.022,
            "confirm_gap": 0.016,
            "repair_gap": 0.010,
            "uncertainty_gate": 0.16,
            "max_confirmations": 3,
            "max_repair_children": 1,
            "noise_policy": "fresh_noise",
            "strength": 0.75,
            "tail_top_time": 1.0,
            "tail_rival_time": 1.0,
            "stop_spend_floor": 62,
        }

    def _commit_stage(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | str],
        initial_budget: int,
    ) -> list[int]:
        ranked = self._rank_particle_ids_from_state(env, particle_ids)
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

    def _late_confirm(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | str],
        initial_budget: int,
    ) -> None:
        previews = self._rank_previews(env, particle_ids)
        if len(previews) < 2:
            return

        best = previews[0]
        runner_up = previews[1]
        gap = float(best.score or 0.0) - float(runner_up.score or 0.0)
        uncertainty = max(float(best.uncertainty or 0.0), float(runner_up.uncertainty or 0.0))
        if gap > float(schedule["confirm_gap"]) and uncertainty < float(schedule["uncertainty_gate"]):
            return

        confirm_ids = [preview.particle_id for preview in previews[: int(schedule["max_confirmations"])]]
        for particle_id in confirm_ids:
            if not self._can_spend(env, initial_budget, schedule, 1):
                return
            try:
                self._preview(env, particle_id)
            except (BudgetExceededError, InvalidActionError):
                return

    def _selective_repair(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | str],
        initial_budget: int,
    ) -> list[int]:
        if int(schedule["max_repair_children"]) <= 0:
            return particle_ids

        previews = self._rank_previews(env, particle_ids)
        if len(previews) < 2:
            return particle_ids

        best = previews[0]
        runner_up = previews[1]
        gap = float(best.score or 0.0) - float(runner_up.score or 0.0)
        uncertainty = max(float(best.uncertainty or 0.0), float(runner_up.uncertainty or 0.0))
        if gap > float(schedule["repair_gap"]) and uncertainty < float(schedule["uncertainty_gate"]):
            return particle_ids

        anchor = runner_up if float(runner_up.uncertainty or 0.0) >= float(best.uncertainty or 0.0) else best
        child_cost = self._child_eval_cost(
            env,
            float(schedule["repair_time"]),
            float(schedule["repair_eval_time"]),
        )
        if not self._can_spend(env, initial_budget, schedule, child_cost):
            return particle_ids

        try:
            child_ids = env.backward(
                anchor.id,
                target_time=float(schedule["repair_time"]),
                noise_policy=str(schedule["noise_policy"]),
                num_children=int(schedule["max_repair_children"]),
                strength=float(schedule["strength"]),
            )
        except (BudgetExceededError, InvalidActionError):
            return particle_ids

        new_ids: list[int] = []
        for child_id in child_ids:
            try:
                self._forward_to(env, child_id, float(schedule["repair_eval_time"]), solver="euler")
                self._preview(env, child_id)
                new_ids.append(child_id)
            except (BudgetExceededError, InvalidActionError):
                break
        return particle_ids + new_ids

    def _tail_stage(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | str],
        initial_budget: int,
    ) -> None:
        ranked = self._rank_particle_ids_from_state(env, particle_ids)
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

        if len(ranked) < 2 or float(schedule["tail_rival_time"]) <= 0.0:
            return

        updated = self._rank_previews(env, particle_ids)
        if len(updated) < 2:
            return
        best = updated[0]
        runner_up = updated[1]
        gap = float(best.score or 0.0) - float(runner_up.score or 0.0)
        uncertainty = max(float(best.uncertainty or 0.0), float(runner_up.uncertainty or 0.0))
        if gap > float(schedule["confirm_gap"]) and uncertainty < float(schedule["uncertainty_gate"]):
            return

        rival_cost = self._segment_cost(env, runner_up.particle_id, float(schedule["tail_rival_time"])) + 1
        if not self._can_spend(env, initial_budget, schedule, rival_cost):
            return
        try:
            self._forward_to(env, runner_up.particle_id, float(schedule["tail_rival_time"]), solver="euler")
            self._preview(env, runner_up.particle_id)
        except (BudgetExceededError, InvalidActionError):
            return

    def _fill_target(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        schedule: dict[str, float | int | str],
        initial_budget: int,
    ) -> None:
        while True:
            candidate_id, next_time = self._best_next_step(env, particle_ids)
            if candidate_id is None or next_time is None:
                return
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
                    and gap > 0.5 * float(schedule["confirm_gap"])
                    and uncertainty < float(schedule["uncertainty_gate"])
                ):
                    return
            if not self._can_spend(env, initial_budget, schedule, 2):
                return
            try:
                self._forward_to(env, candidate_id, next_time, solver="euler")
                self._preview(env, candidate_id)
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
        survivors = []
        for preview in ranked_previews:
            score = float(preview.score or 0.0)
            if len(survivors) < keep or score >= cutoff - margin:
                survivors.append(preview.particle_id)
        return survivors

    def _rank_particle_ids_from_state(self, env: FlowTTSEnv, particle_ids: list[int]) -> list[int]:
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
        previews = []
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

    def _best_next_step(self, env: FlowTTSEnv, particle_ids: list[int]) -> tuple[int | None, float | None]:
        state = env.get_state()
        for particle_id in self._rank_particle_ids_from_state(env, particle_ids):
            particle = state.particles.get(particle_id)
            if particle is None or particle.status != "active":
                continue
            for time_value in env.time_grid:
                if time_value > particle.time + 1e-9:
                    return particle_id, float(time_value)
        return None, None

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
        schedule: dict[str, float | int | str],
        extra_cost: int,
    ) -> bool:
        if extra_cost < 0:
            return False
        spent = self._spent(env, initial_budget)
        limit = min(int(schedule["target_nfe"]), int(env.budget))
        return env.budget_left >= extra_cost and spent + extra_cost <= limit

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

    def _sde_cfg(self, schedule: dict[str, float | int | str]) -> dict[str, float | str]:
        return {
            "noise_scale": float(schedule["sde_noise_scale"]),
            "sigma_max": 1.25,
            "min_time": 0.02,
        }
