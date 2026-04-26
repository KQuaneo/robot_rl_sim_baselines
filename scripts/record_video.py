#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, SAC, TD3

from env_utils import resolve_env_id


ALGOS = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGOS.keys(), required=True)
    parser.add_argument("--env", default="Reacher-v5")
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--video-dir", type=Path, default=Path("videos"))
    args = parser.parse_args()

    env_id = resolve_env_id(args.env)
    model = ALGOS[args.algo].load(args.model)
    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(args.video_dir),
        episode_trigger=lambda episode_id: True,
        name_prefix=f"{env_id}_{args.algo}",
    )

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
    env.close()
    print(f"saved videos under: {args.video_dir}")


if __name__ == "__main__":
    main()
