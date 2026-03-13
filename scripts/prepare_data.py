
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from configs import load_config
from src.data.prepare import prepare_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = prepare_data(cfg)

    print("\n── Done ──────────────────────────────")
    print(f"  Pairs      : {len(data['pairs'])}")
    print(f"  Vocab size : {len(data['vocab'])}")
    print(f"  Mean       : {data['norm_mean'].tolist()}")
    print(f"  Std        : {data['norm_std'].tolist()}")


if __name__ == "__main__":
    main()