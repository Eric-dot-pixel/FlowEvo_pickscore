"""Evaluate the budget-64 best-of-4 deterministic ODE baseline on PickScore SD3.5."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from flow_autotts.eval.discovery import build_round_result
from flow_autotts.eval.metrics import event_log_to_dicts
from flow_autotts.experiments.pickscore_sd35.dataset import sample_prompt_file
from flow_autotts.experiments.pickscore_sd35.env import SD35EnvConfig, SD35PickScoreEnv, SD35Resources
from flow_autotts.experiments.pickscore_sd35.harness import (
    _default_dataset_dir,
    _default_device,
    _default_model_path,
    _default_pickscore_path,
    _write_json,
    compact_summary,
)
from flow_autotts.experiments.pickscore_sd35.merge_shards import merge_histories


REPO_ROOT = Path(__file__).resolve().parents[3]
_BETA_TO_SINGLE_ODE_STEPS = {
    0.0: 2,
    0.25: 5,
    0.5: 9,
    0.75: 12,
    1.0: 16,
}


def _nearest_beta(beta: float) -> float:
    beta = float(beta)
    return min(_BETA_TO_SINGLE_ODE_STEPS, key=lambda item: abs(item - beta))


def _single_root_steps(beta: float) -> int:
    return _BETA_TO_SINGLE_ODE_STEPS[_nearest_beta(beta)]


def _root_seed(sample_seed: int, root_index: int) -> int:
    return (int(sample_seed) * 1_000_003 + int(root_index) * 104_729) % (2**31 - 1)


def _run_one_root(
    *,
    resources: SD35Resources,
    prompt: str,
    sample_seed: int,
    root_index: int,
    single_ode_steps: int,
    env_config: SD35EnvConfig,
) -> dict[str, Any]:
    env = SD35PickScoreEnv(
        resources=resources,
        prompt=prompt,
        seed=_root_seed(sample_seed, root_index),
        budget=single_ode_steps,
        config=env_config,
    )
    particle_id = env.spawn(1)[0]
    for step in range(1, single_ode_steps + 1):
        env.forward(
            particle_id,
            target_time=float(step) / float(single_ode_steps),
            solver="euler",
        )
    answer = env.answer(rule="latest_active")
    return {
        "root_index": int(root_index),
        "reward": answer.reward,
        "score_dict": dict(answer.score_dict),
        "event_log": event_log_to_dicts(answer.event_log),
    }


def _evaluate_sample_bestof4(
    *,
    resources: SD35Resources,
    prompt: str,
    sample_seed: int,
    sample_rank: int,
    prompt_index: int,
    beta: float,
    env_config: SD35EnvConfig,
) -> dict[str, Any]:
    single_ode_steps = _single_root_steps(beta)
    roots = [
        _run_one_root(
            resources=resources,
            prompt=prompt,
            sample_seed=sample_seed,
            root_index=root_index,
            single_ode_steps=single_ode_steps,
            env_config=env_config,
        )
        for root_index in range(4)
    ]
    best = max(
        roots,
        key=lambda item: float(item["reward"]) if item["reward"] is not None else float("-inf"),
    )
    total_nfe = 4 * single_ode_steps
    reward = best["reward"]
    reward_per_nfe = None if reward is None or total_nfe <= 0 else float(reward) / float(total_nfe)
    return {
        "sample_rank": int(sample_rank),
        "prompt_index": int(prompt_index),
        "prompt": prompt,
        "seed": int(sample_seed),
        "answer": {
            "particle_id": None,
            "preview_id": None,
            "reward": reward,
            "nfe_used": total_nfe,
            "rule": "best_of_4_final_reward",
            "score_dict": dict(best["score_dict"]),
        },
        "metrics": {
            "final_reward": reward,
            "nfe": total_nfe,
            "reward_per_nfe": reward_per_nfe,
            "preview_calls": 0,
            "backward_calls": 0,
            "num_particles_spawned": 4,
            "action_counts": {
                "SPAWN": 4,
                "FORWARD": total_nfe,
                "ANSWER": 4,
            },
            "preview_final_correlation": None,
            "false_prune_rate": None,
            "wasted_nfe_rate": None,
        },
        "event_log": [],
        "roots": roots,
    }


def _aggregate_bestof4_actions(episodes: list[dict[str, Any]]) -> dict[str, float]:
    if not episodes:
        return {}
    total_nfe = sum(int(episode["metrics"]["nfe"]) for episode in episodes)
    return {
        "spawn": 4.0,
        "forward": total_nfe / len(episodes),
        "mean_nfe": total_nfe / len(episodes),
    }


def run_bestof4_ode_eval(
    *,
    dataset_dir: str | Path | None = None,
    split: str = "test",
    sample_size: int = 2048,
    sample_seed: int = 42,
    num_shards: int = 1,
    shard_index: int = 0,
    betas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    budget: int = 64,
    output: str | Path | None = None,
    summary_output: str | Path | None = None,
    model_path: str | Path | None = None,
    pickscore_model_path: str | Path | None = None,
    pickscore_processor_path: str | Path | None = None,
    device: str | None = None,
    text_encoder_device: str | None = None,
    offload_text_encoders_after_encode: bool = False,
    score_device: str | None = None,
    dtype: str | None = None,
    score_dtype: str = "float32",
    resolution: int = 512,
    num_steps: int = 10,
    guidance_scale: float = 4.5,
    noise_level: float = 0.7,
    sde_type: str = "sde",
    local_files_only: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    dataset = Path(dataset_dir) if dataset_dir is not None else _default_dataset_dir()
    model = Path(model_path) if model_path is not None else _default_model_path()
    pickscore_model = (
        Path(pickscore_model_path)
        if pickscore_model_path is not None
        else _default_pickscore_path()
    )
    pickscore_processor = (
        Path(pickscore_processor_path)
        if pickscore_processor_path is not None
        else pickscore_model
    )
    runtime_device = device or _default_device()
    runtime_dtype = dtype or ("bfloat16" if runtime_device.startswith("cuda") else "float32")

    all_samples = sample_prompt_file(
        dataset_dir=dataset,
        split=split,
        sample_size=sample_size,
        seed=sample_seed,
    )
    sample_ranks = list(range(len(all_samples)))
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if num_shards > 1:
        ranked = [
            (rank, sample)
            for rank, sample in enumerate(all_samples)
            if rank % num_shards == shard_index
        ]
        sample_ranks = [rank for rank, _sample in ranked]
        samples = [sample for _rank, sample in ranked]
    else:
        samples = all_samples

    env_config = SD35EnvConfig(
        resolution=resolution,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        noise_level=noise_level,
        sde_type=sde_type,
    )
    resources = SD35Resources.load(
        model_path=model,
        pickscore_model_path=pickscore_model,
        pickscore_processor_path=pickscore_processor,
        device=runtime_device,
        text_encoder_device=text_encoder_device,
        offload_text_encoders_after_encode=offload_text_encoders_after_encode,
        score_device=score_device,
        dtype=runtime_dtype,
        score_dtype=score_dtype,
        num_steps=num_steps,
        local_files_only=local_files_only,
        progress=progress,
    )

    history: dict[str, Any] = {
        "experiment": "pickscore_sd35",
        "dataset": str(dataset),
        "split": split,
        "sample_size": sample_size,
        "evaluated_sample_size": len(samples),
        "sample_seed": sample_seed,
        "num_shards": int(num_shards),
        "shard_index": int(shard_index),
        "prompt_sample": [sample.to_dict() for sample in all_samples],
        "evaluated_prompt_sample": [sample.to_dict() for sample in samples],
        "model_path": str(model),
        "pickscore_model_path": str(pickscore_model),
        "device": runtime_device,
        "text_encoder_device": text_encoder_device or runtime_device,
        "offload_text_encoders_after_encode": bool(offload_text_encoders_after_encode),
        "score_device": score_device or runtime_device,
        "dtype": runtime_dtype,
        "betas": [float(beta) for beta in betas],
        "budget": int(budget),
        "env_config": asdict(env_config),
        "rounds": [],
    }

    beta_results = []
    for beta in betas:
        episodes = [
            _evaluate_sample_bestof4(
                resources=resources,
                prompt=sample.prompt,
                sample_seed=sample.seed,
                sample_rank=sample_ranks[local_rank],
                prompt_index=sample.index,
                beta=float(beta),
                env_config=env_config,
            )
            for local_rank, sample in enumerate(samples)
        ]
        rewards = [
            episode["metrics"]["final_reward"]
            for episode in episodes
            if episode["metrics"]["final_reward"] is not None
        ]
        nfes = [episode["metrics"]["nfe"] for episode in episodes]
        reward_per_nfes = [
            episode["metrics"]["reward_per_nfe"]
            for episode in episodes
            if episode["metrics"]["reward_per_nfe"] is not None
        ]
        beta_results.append(
            {
                "beta": float(beta),
                "num_samples": len(episodes),
                "final_reward": mean(rewards) if rewards else None,
                "nfe": mean(nfes) if nfes else 0.0,
                "reward_per_nfe": mean(reward_per_nfes) if reward_per_nfes else None,
                "episodes": episodes,
                "action_statistics": _aggregate_bestof4_actions(episodes),
            }
        )
        resources.prompt_cache.clear()
        if str(resources.device).startswith("cuda") and hasattr(resources.torch, "cuda"):
            resources.torch.cuda.empty_cache()

    round_result = build_round_result(
        round_id=0,
        controller_name="BestOf4DeterministicODEBaseline",
        beta_sweep_results=beta_results,
    )
    round_result["controller_key"] = "bestof4_ode_baseline"
    history["rounds"].append(round_result)

    if output is not None:
        _write_json(history, output)
    summary = compact_summary(history)
    if summary_output is not None:
        _write_json(summary, summary_output)
    return history


def _split_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.replace(",", " ").split()]
    return [device for device in devices if device]


def _optional_device(devices: Sequence[str], index: int) -> str | None:
    if not devices:
        return None
    if len(devices) == 1:
        return devices[0]
    return devices[index]


def run_parallel_bestof4_ode_eval(args: argparse.Namespace) -> dict[str, Any]:
    import subprocess
    import sys

    devices = _split_devices(args.devices)
    if not devices:
        raise ValueError("--devices must contain at least one device")
    text_devices = _split_devices(args.text_encoder_devices or "")
    score_devices = _split_devices(args.score_devices or "")
    if text_devices and len(text_devices) not in {1, len(devices)}:
        raise ValueError("--text-encoder-devices must have length 1 or match --devices")
    if score_devices and len(score_devices) not in {1, len(devices)}:
        raise ValueError("--score-devices must have length 1 or match --devices")

    output = Path(args.output).expanduser().resolve()
    summary_output = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output is not None
        else None
    )
    shard_root = (
        Path(args.shard_output_dir).expanduser().resolve()
        if args.shard_output_dir
        else output.parent / "shards"
    )
    shard_root.mkdir(parents=True, exist_ok=True)

    procs = []
    shard_paths = []
    for shard_index, device in enumerate(devices):
        shard_dir = shard_root / f"shard_{shard_index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        history_path = shard_dir / "history.json"
        summary_path = shard_dir / "summary.json"
        shard_paths.append(history_path)
        cmd = [
            sys.executable,
            "-m",
            "flow_autotts.experiments.pickscore_sd35.bestof4_ode_baseline_test",
            "--devices",
            device,
            "--text-encoder-devices",
            _optional_device(text_devices, shard_index) or "",
            "--score-devices",
            _optional_device(score_devices, shard_index) or "",
            "--dataset",
            str(args.dataset),
            "--split",
            str(args.split),
            "--sample-size",
            str(args.sample_size),
            "--sample-seed",
            str(args.sample_seed),
            "--num-shards",
            str(len(devices)),
            "--shard-index",
            str(shard_index),
            "--betas",
            *[str(beta) for beta in args.betas],
            "--budget",
            str(args.budget),
            "--output",
            str(history_path),
            "--summary-output",
            str(summary_path),
            "--model",
            str(args.model),
            "--pickscore-model",
            str(args.pickscore_model),
            "--num-steps",
            str(args.num_steps),
            "--resolution",
            str(args.resolution),
            "--guidance-scale",
            str(args.guidance_scale),
            "--noise-level",
            str(args.noise_level),
            "--sde-type",
            str(args.sde_type),
            "--score-dtype",
            str(args.score_dtype),
            "--device",
            device,
        ]
        if args.pickscore_processor is not None:
            cmd.extend(["--pickscore-processor", str(args.pickscore_processor)])
        if _optional_device(text_devices, shard_index):
            cmd.extend(["--text-encoder-device", _optional_device(text_devices, shard_index)])
        if _optional_device(score_devices, shard_index):
            cmd.extend(["--score-device", _optional_device(score_devices, shard_index)])
        if args.dtype is not None:
            cmd.extend(["--dtype", str(args.dtype)])
        if args.allow_remote_files:
            cmd.append("--allow-remote-files")
        if args.progress:
            cmd.append("--progress")
        if args.offload_text_encoders_after_encode:
            cmd.append("--offload-text-encoders-after-encode")

        (shard_dir / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
        stdout = (shard_dir / "stdout.log").open("w", encoding="utf-8")
        stderr = (shard_dir / "stderr.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True)
        stdout.close()
        stderr.close()
        procs.append((shard_index, device, shard_dir, proc))

    failures = []
    for shard_index, device, shard_dir, proc in procs:
        returncode = proc.wait()
        if returncode != 0:
            failures.append(f"shard {shard_index} on {device}: rc={returncode}, dir={shard_dir}")
    if failures:
        raise RuntimeError("; ".join(failures))

    merged = merge_histories(shard_paths, output)
    summary = compact_summary(merged)
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output": str(output),
        "summary_output": str(summary_output) if summary_output is not None else "",
        "shard_root": str(shard_root),
        "devices": devices,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", required=True, help="Comma or space separated device list")
    parser.add_argument("--text-encoder-devices", default="")
    parser.add_argument("--score-devices", default="")
    parser.add_argument("--shard-output-dir", default=None)
    parser.add_argument("--dataset", default=str(_default_dataset_dir()))
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-size", type=int, default=2048)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--model", default=str(_default_model_path()))
    parser.add_argument("--pickscore-model", default=str(_default_pickscore_path()))
    parser.add_argument("--pickscore-processor", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--text-encoder-device", default=None)
    parser.add_argument("--offload-text-encoders-after-encode", action="store_true")
    parser.add_argument("--score-device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--score-dtype", default="float32")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--noise-level", type=float, default=0.7)
    parser.add_argument("--sde-type", choices=["sde", "cps"], default="sde")
    parser.add_argument("--allow-remote-files", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    if args.parallel:
        result = run_parallel_bestof4_ode_eval(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    history = run_bestof4_ode_eval(
        dataset_dir=args.dataset,
        split=args.split,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        betas=args.betas,
        budget=args.budget,
        output=args.output,
        summary_output=args.summary_output,
        model_path=args.model,
        pickscore_model_path=args.pickscore_model,
        pickscore_processor_path=args.pickscore_processor,
        device=args.device,
        text_encoder_device=args.text_encoder_device,
        offload_text_encoders_after_encode=args.offload_text_encoders_after_encode,
        score_device=args.score_device,
        dtype=args.dtype,
        score_dtype=args.score_dtype,
        resolution=args.resolution,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        noise_level=args.noise_level,
        sde_type=args.sde_type,
        local_files_only=not args.allow_remote_files,
        progress=args.progress,
    )
    print(json.dumps(compact_summary(history), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
