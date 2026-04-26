#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def moving_average(values, window: int):
    return pd.Series(values).rolling(window=window, min_periods=1).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="CSV files in label=path format")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plt.figure(figsize=(8, 4.5))
    for item in args.input:
        label, path = item.split("=", 1)
        df = pd.read_csv(path)
        plt.plot(df["timesteps"], moving_average(df["episode_return"], args.window), label=label)

    plt.xlabel("Timesteps")
    plt.ylabel("Episode Return")
    plt.legend()
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=160)
    print(f"saved comparison plot: {args.output}")


if __name__ == "__main__":
    main()
