#!/usr/bin/env python3
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
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
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.csv"))
    args = parser.parse_args()

    env_id = resolve_env_id(args.env)
    model = ALGOS[args.algo].load(args.model)
    rows = []
    for episode in range(args.episodes):
        env = gym.make(env_id)
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        truncated = False
        episode_return = 0.0
        episode_length = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_length += 1
        env.close()
        rows.append(
            {
                "env": env_id,
                "algorithm": args.algo,
                "episode": episode,
                "return": episode_return,
                "length": episode_length,
            }
        )

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df)
    print(f"mean={df['return'].mean():.3f}, std={df['return'].std(ddof=0):.3f}")
    print(f"saved evaluation: {args.output}")


if __name__ == "__main__":
    main()
