"""Utility script to evaluate CDVIB beta settings on CIFAR-10."""

import argparse
import csv
import shlex
import statistics
from pathlib import Path
from typing import Dict, List

import main as sqvae_main


BASE_DIR = Path(__file__).resolve().parent


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
        default=str(BASE_DIR / "experiments" / "cifar10_cdvib_beta_results.csv"),
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


def _build_main_argv(
    config: str, seed: int, beta: float, gpu: str, extra_args: str
) -> List[str]:
    argv: List[str] = ["-c", config, "--save", "--seed", str(seed), "--cdvib_beta", str(beta)]
    if gpu is not None:
        argv.extend(["--gpu", gpu])
    if extra_args:
        argv.extend(shlex.split(extra_args))
    return argv


def _extract_test_reconstruction(test_result: Dict[str, float], beta: float, seed: int) -> float:
    if "mse" in test_result:
        return float(test_result["mse"])
    if "loss" in test_result:
        return float(test_result["loss"])
    raise KeyError(
        "Test result does not contain a reconstruction-compatible metric for "
        f"seed={seed}, beta={beta}."
    )


def run_single(config: str, seed: int, beta: float, gpu: str, extra_args: str) -> float:
    argv = _build_main_argv(config, seed, beta, gpu, extra_args)
    parser = sqvae_main.build_arg_parser()
    args = parser.parse_args(argv)
    run_info = sqvae_main.run_experiment(args)

    checkpoint_path = Path(run_info["checkpoint_dir"])
    plots_path = checkpoint_path / "plots.npy"
    if not plots_path.exists():
        raise FileNotFoundError(
            f"Could not find plots.npy at {plots_path} for seed={seed}, beta={beta}."
        )

    test_result = run_info["test_result"]
    test_reconst = _extract_test_reconstruction(test_result, beta, seed)
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
