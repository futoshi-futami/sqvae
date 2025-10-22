"""Utility script to evaluate CDVIB beta settings on Fashion-MNIST."""

import argparse
from pathlib import Path
from typing import Dict, List

from run_cdvib_beta_sweep import BASE_DIR, print_summary, run_single, save_results


DEFAULT_CONFIG = "fashion-mnist_gauss_1.yaml"
DEFAULT_OUTPUT = BASE_DIR / "experiments" / "fashion_mnist_cdvib_beta_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Fashion-MNIST SQVAE experiments across multiple seeds and CDVIB betas"
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="configuration file to pass to main.py",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="number of seeds starting from zero",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.0, 1e-3],
        help="list of CDVIB beta values to sweep",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU index passed to main.py (use empty string for CPU)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="CSV file used to store per-seed reconstruction losses",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help=(
            "additional arguments forwarded to main.py as a raw string "
            "(e.g. '--dbg')"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: Dict[float, List[dict]] = {}

    for seed in range(args.seeds):
        for beta in args.betas:
            print(f"Running seed={seed}, cdvib_beta={beta}...")
            test_reconst = run_single(
                args.config, seed, beta, args.gpu, args.extra_args
            )
            print(
                f"  -> test reconstruction loss: {test_reconst:.6f}",
                flush=True,
            )
            results.setdefault(beta, []).append(
                {"seed": seed, "test_reconstruction": test_reconst}
            )

    output_path = Path(args.output)
    save_results(output_path, results)
    print(f"\nSaved per-run results to {output_path}")
    print_summary(results)


if __name__ == "__main__":
    main()
