"""Filter PickScore visual-compare samples where controller beats ODE baseline."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DIR = (
    Path("logs")
    / "flow_autotts"
    / "pickscore_sd35"
    / "r0004_vs_ode_b64_beta1_visual_compare_test"
    / "samples"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reward(metadata: dict[str, Any], key: str) -> float | None:
    output = (metadata.get("outputs") or {}).get(key) or {}
    value = output.get("reward")
    if value is None:
        return None
    return float(value)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def collect_wins(source_dir: Path) -> list[dict[str, Any]]:
    wins: list[dict[str, Any]] = []
    for sample_dir in sorted(source_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.is_file():
            continue

        metadata = _read_json(metadata_path)
        controller_reward = _reward(metadata, "controller")
        baseline_reward = _reward(metadata, "ode_b64")
        if controller_reward is None or baseline_reward is None:
            continue

        delta = controller_reward - baseline_reward
        if delta <= 0:
            continue

        wins.append(
            {
                "source_dir": str(sample_dir),
                "sample_dir_name": sample_dir.name,
                "sample_rank": int(metadata.get("sample_rank", -1)),
                "prompt_index": int(metadata.get("prompt_index", -1)),
                "prompt": str(metadata.get("prompt", "")),
                "seed": int(metadata.get("seed", -1)),
                "controller_reward": controller_reward,
                "baseline_reward": baseline_reward,
                "reward_delta": delta,
            }
        )
    wins.sort(key=lambda item: (float(item["reward_delta"]), float(item["controller_reward"])), reverse=True)
    return wins


def copy_ranked_wins(wins: list[dict[str, Any]], output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(wins, start=1):
        source = Path(str(item["source_dir"]))
        delta = float(item["reward_delta"])
        target_name = (
            f"{index:05d}_delta_{delta:+.6f}_"
            f"ctrl_{float(item['controller_reward']):.6f}_"
            f"base_{float(item['baseline_reward']):.6f}_"
            f"{_safe_name(str(item['sample_dir_name']))}"
        )
        shutil.copytree(source, output_dir / target_name)


def write_summary(wins: list[dict[str, Any]], output_dir: Path, source_dir: Path) -> None:
    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "num_controller_wins": len(wins),
        "items": wins,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "rank\treward_delta\tcontroller_reward\tbaseline_reward\tsample_rank\tprompt_index\tsource_dir\tprompt"
    ]
    for rank, item in enumerate(wins, start=1):
        prompt = str(item["prompt"]).replace("\t", " ").replace("\n", " ")
        lines.append(
            "\t".join(
                [
                    str(rank),
                    f"{float(item['reward_delta']):.9f}",
                    f"{float(item['controller_reward']):.9f}",
                    f"{float(item['baseline_reward']):.9f}",
                    str(item["sample_rank"]),
                    str(item["prompt_index"]),
                    str(item["source_dir"]),
                    prompt,
                ]
            )
        )
    (output_dir / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory not found: {source_dir}")

    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else source_dir.parent / "controller_wins_by_reward_delta"
    )

    wins = collect_wins(source_dir)
    copy_ranked_wins(wins, output_dir)
    write_summary(wins, output_dir, source_dir)
    print(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "output_dir": str(output_dir),
                "num_controller_wins": len(wins),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
