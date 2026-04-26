#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def moving_average(values, window: int):
    return pd.Series(values).rolling(window=window, min_periods=1).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    output = args.output or args.csv.with_suffix(".png")

    plt.figure(figsize=(8, 4.5))
    plt.plot(df["timesteps"], df["episode_return"], alpha=0.25, label="episode return")
    plt.plot(df["timesteps"], moving_average(df["episode_return"], args.window), label=f"MA-{args.window}")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Return")
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    print(f"saved plot: {output}")


if __name__ == "__main__":
    main()
