from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from flow_autotts.workflow.runner import (
    _maybe_early_stop,
    _promotion_decision,
    _target_nfe_for_beta,
    archive_round,
    build_context_pack,
    parse_round_id,
    resolve_method_template,
)


class WorkflowTests(unittest.TestCase):
    def test_parse_round_id(self):
        self.assertEqual(
            parse_round_id("r0003_20260513_120102_abcdef12"),
            (3, "20260513_120102", "abcdef12"),
        )
        self.assertIsNone(parse_round_id("round3"))

    def test_resolve_suffix_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            method = root / "pkg" / "optimal.py"
            template = root / "pkg" / "optimal.template.py"
            method.parent.mkdir()
            method.write_text("# method\n", encoding="utf-8")
            template.write_text("# template\n", encoding="utf-8")

            self.assertEqual(
                resolve_method_template(root, "pkg/optimal.py", None),
                template.resolve(),
            )

    def test_archive_round_copies_method_results_and_clears_source_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            method = root / "flow_autotts" / "controllers" / "optimal.py"
            method.parent.mkdir(parents=True)
            method.write_text("class OptimalController: pass\n", encoding="utf-8")
            result_dir = root / "training_results"
            result_dir.mkdir()
            (result_dir / "summary.json").write_text("{}", encoding="utf-8")

            dest = archive_round(
                workdir=root,
                history_dir="history",
                round_id="r0000_20260513_120102_abcdef12",
                method_file="flow_autotts/controllers/optimal.py",
                method_src=method,
                result_dir=result_dir,
                dest_allow_exists=False,
            )

            self.assertTrue((dest / "flow_autotts" / "controllers" / "optimal.py").is_file())
            self.assertTrue((dest / "proposal_results" / "summary.json").is_file())
            self.assertEqual(list(result_dir.iterdir()), [])

    def test_context_pack_points_proposer_at_narrow_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_round = root / "history" / "r0000_20260513_120102_abcdef12"
            summary = history_round / "proposal_results" / "summary.json"
            snapshot = history_round / "flow_autotts" / "controllers" / "optimal.py"
            summary.parent.mkdir(parents=True)
            snapshot.parent.mkdir(parents=True)
            summary.write_text(
                """
{
      "rounds": [
    {
      "beta_sweep": [
        {
          "beta": 0.5,
          "nfe": 37.0,
          "reward": 0.84,
          "action_statistics": {"spawn": 3, "preview": 4, "backward": 1, "mean_nfe": 37}
        }
      ]
    }
  ]
}
""".strip(),
                encoding="utf-8",
            )
            snapshot.write_text("class OptimalController: pass\n", encoding="utf-8")
            baseline = (
                root
                / "logs"
                / "flow_autotts"
                / "pickscore_sd35"
                / "ode_baseline_equiv"
                / "aggregate_summary.json"
            )
            baseline.parent.mkdir(parents=True)
            baseline.write_text(
                '[{"beta": 0.5, "nfe": 36, "reward": 0.83, "num_samples": 500, "behavior_summary": "best-of-4 deterministic ODE"}]',
                encoding="utf-8",
            )
            out = root / "logs"

            old_baseline = os.environ.get("WORKFLOW_BASELINE_SUMMARY")
            try:
                os.environ["WORKFLOW_BASELINE_SUMMARY"] = str(baseline)
                context = build_context_pack(
                    workdir=root,
                    method_file="flow_autotts/controllers/optimal.py",
                    history_dir="history",
                    template_path=None,
                    output_dir=out,
                    max_history_rounds=1,
                )
            finally:
                if old_baseline is None:
                    os.environ.pop("WORKFLOW_BASELINE_SUMMARY", None)
                else:
                    os.environ["WORKFLOW_BASELINE_SUMMARY"] = old_baseline
            text = context.read_text(encoding="utf-8")

            self.assertIn("Allowed First-Pass Reads", text)
            self.assertIn("Edit only `flow_autotts/controllers/optimal.py`", text)
            self.assertIn("r0000_20260513_120102_abcdef12", text)
            self.assertIn("Do not bulk-read raw `history.json`", text)
            self.assertIn("Baseline References", text)
            self.assertIn("Beta Target Curve", text)
            self.assertIn("Action Semantics And Likely Effects", text)
            self.assertIn("Recent Round Frontier Comparison", text)
            self.assertIn("Beta Opportunities", text)
            self.assertIn("Historical Action Effects", text)
            self.assertIn("Historical Best Near Beta Target", text)
            self.assertIn("Regression Ledger", text)
            self.assertIn(
                "| 0.500 | 36.000 | 0.830000 | best-of-4 deterministic ODE |",
                text,
            )
            self.assertIn(
                "| r0000 | 0.500 | 37.000 | 36.000 | 1.000 | over +1.0 | 0.840000 | 0.830000 | 0.010000 | 36.000 | 0.010000 | spawn=3.00, preview=4.00, backward=1.00, mean_nfe=37.00 |",
                text,
            )

    def test_beta_zero_target_is_locked_to_ten_nfe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = (
                root
                / "logs"
                / "flow_autotts"
                / "pickscore_sd35"
                / "ode_baseline_equiv"
                / "aggregate_summary.json"
            )
            baseline.parent.mkdir(parents=True)
            baseline.write_text(
                '[{"beta": 0.0, "nfe": 8, "reward": 0.67, "behavior_summary": "best-of-4 deterministic ODE"}]',
                encoding="utf-8",
            )
            out = root / "logs"

            old_baseline = os.environ.get("WORKFLOW_BASELINE_SUMMARY")
            try:
                os.environ["WORKFLOW_BASELINE_SUMMARY"] = str(baseline)
                context = build_context_pack(
                    workdir=root,
                    method_file="flow_autotts/controllers/optimal.py",
                    history_dir="history",
                    template_path=None,
                    output_dir=out,
                    max_history_rounds=0,
                )
            finally:
                if old_baseline is None:
                    os.environ.pop("WORKFLOW_BASELINE_SUMMARY", None)
                else:
                    os.environ["WORKFLOW_BASELINE_SUMMARY"] = old_baseline
            text = context.read_text(encoding="utf-8")

            self.assertIn(
                "| 0.000 | 10.000 | 0.670000 | best-of-4 deterministic ODE |",
                text,
            )

    def test_fixed_target_map_matches_experiment_schedule(self):
        self.assertEqual(_target_nfe_for_beta(0.0), 10.0)
        self.assertEqual(_target_nfe_for_beta(0.25), 20.0)
        self.assertEqual(_target_nfe_for_beta(0.5), 36.0)
        self.assertEqual(_target_nfe_for_beta(0.75), 48.0)
        self.assertEqual(_target_nfe_for_beta(1.0), 64.0)

    def test_promotion_decision_rejects_worse_candidate(self):
        baseline_rows = [
            {"beta": 0.0, "nfe": 8, "reward": 0.67},
            {"beta": 0.25, "nfe": 20, "reward": 0.80},
            {"beta": 0.5, "nfe": 36, "reward": 0.83},
            {"beta": 0.75, "nfe": 48, "reward": 0.84},
            {"beta": 1.0, "nfe": 64, "reward": 0.845},
        ]
        incumbent_summary = {
            "rounds": [
                {
                    "beta_sweep": [
                        {"beta": 0.0, "nfe": 10, "reward": 0.75},
                        {"beta": 0.25, "nfe": 20, "reward": 0.81},
                        {"beta": 0.5, "nfe": 36, "reward": 0.835},
                        {"beta": 0.75, "nfe": 48, "reward": 0.841},
                        {"beta": 1.0, "nfe": 64, "reward": 0.846},
                    ]
                }
            ]
        }
        candidate_summary = {
            "rounds": [
                {
                    "beta_sweep": [
                        {"beta": 0.0, "nfe": 10, "reward": 0.74},
                        {"beta": 0.25, "nfe": 20, "reward": 0.805},
                        {"beta": 0.5, "nfe": 36, "reward": 0.834},
                        {"beta": 0.75, "nfe": 48, "reward": 0.8405},
                        {"beta": 1.0, "nfe": 64, "reward": 0.8455},
                    ]
                }
            ]
        }
        status, reason, _candidate_score, _incumbent_score = _promotion_decision(
            candidate_summary=candidate_summary,
            incumbent_summary=incumbent_summary,
            baseline_rows=baseline_rows,
        )
        self.assertEqual(status, "rejected")
        self.assertIn("did not beat incumbent", reason)

    def test_context_pack_includes_rejected_round_lessons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history = root / "history"
            history.mkdir()

            baseline = (
                root
                / "logs"
                / "flow_autotts"
                / "pickscore_sd35"
                / "ode_baseline_equiv"
                / "aggregate_summary.json"
            )
            baseline.parent.mkdir(parents=True)
            baseline.write_text(
                """
[
  {"beta": 0.0, "nfe": 8, "reward": 0.67, "behavior_summary": "ode"},
  {"beta": 0.25, "nfe": 20, "reward": 0.80, "behavior_summary": "ode"},
  {"beta": 0.5, "nfe": 36, "reward": 0.83, "behavior_summary": "ode"},
  {"beta": 0.75, "nfe": 48, "reward": 0.84, "behavior_summary": "ode"},
  {"beta": 1.0, "nfe": 64, "reward": 0.845, "behavior_summary": "ode"}
]
""".strip(),
                encoding="utf-8",
            )

            accepted = history / "r0000_20260513_120102_abcdef12"
            rejected = history / "r0001_20260513_120102_abcdef12"
            for round_dir, reward_025, preview_025 in (
                (accepted, 0.81, 2.0),
                (rejected, 0.79, 5.0),
            ):
                summary = round_dir / "proposal_results" / "summary.json"
                snapshot = round_dir / "flow_autotts" / "controllers" / "optimal.py"
                summary.parent.mkdir(parents=True)
                snapshot.parent.mkdir(parents=True)
                snapshot.write_text("class OptimalController: pass\n", encoding="utf-8")
                summary.write_text(
                    f"""
{{
  "rounds": [
    {{
      "beta_sweep": [
        {{
          "beta": 0.25,
          "nfe": 20.0,
          "reward": {reward_025},
          "action_statistics": {{
            "spawn": 1.0,
            "forward": 4.0,
            "preview": {preview_025},
            "prune": 1.0,
            "mean_nfe": 20.0
          }}
        }}
      ]
    }}
  ]
}}
""".strip(),
                    encoding="utf-8",
                )

            (history / "workflow_index.jsonl").write_text(
                """
{"round_index": 0, "round_id": "r0000_20260513_120102_abcdef12", "promotion_status": "accepted", "incumbent_round_id_after_round": "r0000_20260513_120102_abcdef12"}
{"round_index": 1, "round_id": "r0001_20260513_120102_abcdef12", "promotion_status": "rejected", "promotion_reason": "candidate did not beat incumbent on fixed-target frontier score", "incumbent_round_id_after_round": "r0000_20260513_120102_abcdef12"}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            out = root / "logs_out"
            old_baseline = os.environ.get("WORKFLOW_BASELINE_SUMMARY")
            old_promoted_only = os.environ.get("WORKFLOW_CONTEXT_PROMOTED_ONLY")
            try:
                os.environ["WORKFLOW_BASELINE_SUMMARY"] = str(baseline)
                os.environ["WORKFLOW_CONTEXT_PROMOTED_ONLY"] = "1"
                context = build_context_pack(
                    workdir=root,
                    method_file="flow_autotts/controllers/optimal.py",
                    history_dir="history",
                    template_path=None,
                    output_dir=out,
                    max_history_rounds=2,
                )
            finally:
                if old_baseline is None:
                    os.environ.pop("WORKFLOW_BASELINE_SUMMARY", None)
                else:
                    os.environ["WORKFLOW_BASELINE_SUMMARY"] = old_baseline
                if old_promoted_only is None:
                    os.environ.pop("WORKFLOW_CONTEXT_PROMOTED_ONLY", None)
                else:
                    os.environ["WORKFLOW_CONTEXT_PROMOTED_ONLY"] = old_promoted_only

            text = context.read_text(encoding="utf-8")
            self.assertIn("Rejected Round Lessons", text)
            self.assertIn("Rejected `r0001` vs incumbent `r0000`", text)
            self.assertIn("preview +3.00", text)
            self.assertIn("more preview budget did not translate into better ranking/refinement", text)

    def test_early_stop_triggers_after_three_rejections_post_min_rounds(self):
        results = [
            {"promotion_status": "accepted", "candidate_score": {"total_reward_gap": 0.10, "min_reward_gap": -0.01}},
            {"promotion_status": "accepted", "candidate_score": {"total_reward_gap": 0.101, "min_reward_gap": -0.01}},
            {"promotion_status": "rejected"},
            {"promotion_status": "rejected"},
            {"promotion_status": "rejected"},
        ]
        should_stop, reason = _maybe_early_stop(
            results=results,
            min_rounds_before_stop=5,
            max_consecutive_rejections_before_stop=3,
            min_total_reward_gap_improvement=0.001,
        )
        self.assertTrue(should_stop)
        self.assertIn("last 3 rounds were all rejected", reason)

    def test_early_stop_does_not_trigger_before_min_rounds(self):
        results = [
            {"promotion_status": "accepted", "candidate_score": {"total_reward_gap": 0.10, "min_reward_gap": -0.01}},
            {"promotion_status": "rejected"},
            {"promotion_status": "rejected"},
            {"promotion_status": "rejected"},
        ]
        should_stop, _reason = _maybe_early_stop(
            results=results,
            min_rounds_before_stop=5,
            max_consecutive_rejections_before_stop=3,
            min_total_reward_gap_improvement=0.001,
        )
        self.assertFalse(should_stop)


if __name__ == "__main__":
    unittest.main()
