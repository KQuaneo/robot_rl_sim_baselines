#!/usr/bin/env python3
import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from env_utils import resolve_env_id


ALGOS = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


def make_run_name(algo: str, env_id: str, seed: int, suffix: str | None = None) -> str:
    name = f"{env_id}_{algo}_seed{seed}"
    if suffix:
        name = f"{name}_{suffix}"
    return name


def save_curve_csv(log_dir: Path, output_csv: Path) -> None:
    data = load_results(str(log_dir))
    x, y = ts2xy(data, "timesteps")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text("timesteps,episode_return\n", encoding="utf-8")
    with output_csv.open("a", encoding="utf-8") as f:
        for step, ret in zip(x, y):
            f.write(f"{int(step)},{float(ret)}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGOS.keys(), required=True)
    parser.add_argument("--env", default="Reacher-v5")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--run-name-suffix", default=None)
    parser.add_argument("--train-freq", type=int, default=None)
    parser.add_argument("--gradient-steps", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--progress-bar", action="store_true")
    args = parser.parse_args()

    env_id = resolve_env_id(args.env)
    run_name = make_run_name(args.algo, env_id, args.seed, args.run_name_suffix)
    run_log_dir = args.log_dir / run_name
    run_ckpt_dir = args.checkpoint_dir / run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(gym.make(env_id), str(run_log_dir))
    model_cls = ALGOS[args.algo]
    model_kwargs = {}
    for name in ("train_freq", "gradient_steps", "learning_starts", "buffer_size", "batch_size"):
        value = getattr(args, name)
        if value is not None:
            model_kwargs[name] = value
    model = model_cls(
        "MlpPolicy",
        env,
        verbose=args.verbose,
        tensorboard_log=str(run_log_dir),
        seed=args.seed,
        **model_kwargs,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(run_ckpt_dir),
        name_prefix=run_name,
        save_replay_buffer=args.algo in {"sac", "td3"},
    )
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback, progress_bar=args.progress_bar)

    final_path = run_ckpt_dir / "final_model"
    model.save(final_path)
    save_curve_csv(run_log_dir, args.log_dir / f"{run_name}_returns.csv")
    env.close()
    print(f"saved model: {final_path}.zip")
    print(f"saved returns: {args.log_dir / f'{run_name}_returns.csv'}")


if __name__ == "__main__":
    main()
