"""Utility script to evaluate CDVIB beta settings on CIFAR-10."""

import argparse
import csv
import re
import shlex
import statistics
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CIFAR-10 SQVAE experiments across multiple seeds and CDVIB betas"
        )
    )
    parser.add_argument(
        "--config",
        default="cifar10_gauss_1.yaml",
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
        default=str(ROOT / "experiments" / "cifar10_cdvib_beta_results.csv"),
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


def _extract_checkpoint_path(output: str) -> Path:
    match = re.search(r"\[Checkpoint path\]\s*(.+)", output)
    if not match:
        raise RuntimeError(
            "Failed to locate checkpoint path in main.py output.\n" + output
        )
    return Path(match.group(1).strip())


def run_single(config: str, seed: int, beta: float, gpu: str, extra_args: str) -> float:
    cmd: List[str] = [
        "python",
        "main.py",
        "-c",
        config,
        "--save",
        "--seed",
        str(seed),
        "--cdvib_beta",
        str(beta),
    ]
    if gpu is not None:
        cmd.extend(["--gpu", gpu])
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = "\n".join([result.stdout, result.stderr])
    if result.returncode != 0:
        raise RuntimeError(
            f"Experiment failed for seed={seed}, beta={beta}:\n{combined_output}"
        )

    checkpoint_path = _extract_checkpoint_path(combined_output)
    plots_path = checkpoint_path / "plots.npy"
    if not plots_path.exists():
        raise FileNotFoundError(
            f"Could not find plots.npy at {plots_path} for seed={seed}, beta={beta}."
        )

    plots: Dict[str, List[float]] = np.load(plots_path, allow_pickle=True).item()
    if "mse_test" not in plots or not plots["mse_test"]:
        raise KeyError(
            "plots.npy does not contain mse_test entries required for reconstruction "
            f"metrics at {plots_path}."
        )

    test_reconst = float(plots["mse_test"][-1])
    return test_reconst


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_results(output_path: Path, results: Dict[float, List[dict]]) -> None:
    ensure_directory(output_path)
    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["seed", "cdvib_beta", "test_reconstruction"])
        for beta, entries in sorted(results.items(), key=lambda item: item[0]):
            for entry in entries:
                writer.writerow([entry["seed"], beta, entry["test_reconstruction"]])


def print_summary(results: Dict[float, List[dict]]) -> None:
    header = f'{"beta":>12}{"mean":>15}{"std":>15}'
    print("\nCDVIB beta sweep summary (test reconstruction)\n")
    print(header)
    print("-" * len(header))
    for beta, entries in sorted(results.items(), key=lambda item: item[0]):
        values = [entry["test_reconstruction"] for entry in entries]
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{beta:12.6g}{mean_val:15.6f}{std_val:15.6f}")


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
