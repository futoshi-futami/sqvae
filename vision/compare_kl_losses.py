"""Command-line helper to compare SQ-VAE test losses with and without KL regularization.

The script loads a trained SQ-VAE checkpoint, runs the test evaluation twice (with
and without the index KL penalty), and reports the aggregated metrics so that the
impact of the KL regularizer can be inspected easily.
"""

import argparse
import os
from types import SimpleNamespace
from typing import Dict, Tuple

import torch

from main import load_config
from trainer import GaussianSQVAETrainer, VmfSQVAETrainer
from util import get_loader, set_seeds


def _build_trainer(cfgs, flgs, train_loader, val_loader, test_loader):
    if cfgs.model.name == "GaussianSQVAE":
        return GaussianSQVAETrainer(cfgs, flgs, train_loader, val_loader, test_loader)
    if cfgs.model.name == "VmfSQVAE":
        return VmfSQVAETrainer(cfgs, flgs, train_loader, val_loader, test_loader)
    raise ValueError(f"Unsupported model: {cfgs.model.name}")


def _evaluate(trainer, mode: str = "test") -> Dict[str, float]:
    trainer.model.eval()
    # Match the behaviour of Trainer._test by running both stochastic and
    # deterministic quantization passes before collecting the reported metrics.
    trainer._test_sub(False, mode)
    return trainer._test_sub(True, mode)


def _resolve_quantizer(trainer):
    model = trainer.model
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return model.quantizer


def _format_result(name: str, result: Dict[str, float]) -> str:
    parts = [f"{name} results:"]
    for key in sorted(result.keys()):
        parts.append(f"  {key:>10s}: {result[key]:.6f}")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SQ-VAE reconstruction metrics with/without KL regularization.")
    parser.add_argument("-c", "--config_file", required=True,
                        help="Name of the config file under vision/configs")
    parser.add_argument("-ts", "--timestamp", required=True,
                        help="Timestamp directory of the trained checkpoint")
    parser.add_argument("--gpu", default="0",
                        help="CUDA_VISIBLE_DEVICES value used for evaluation")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed used when initialising loaders (defaults to 0)")
    parser.add_argument("--dbg", action="store_true",
                        help="Print per-epoch logs during evaluation")
    return parser.parse_args()


def main() -> Tuple[Dict[str, float], Dict[str, float]]:
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Reuse the loader configuration logic from main.load_config.
    load_args = SimpleNamespace(
        config_file=args.config_file,
        seed=args.seed,
        save=False,
        dbg=args.dbg,
    )
    cfgs, flgs = load_config(load_args)

    set_seeds(args.seed)
    train_loader, val_loader, test_loader = get_loader(
        cfgs.dataset.name, cfgs.path_dataset, cfgs.train.bs, cfgs.nworker)

    trainer = _build_trainer(cfgs, flgs, train_loader, val_loader, test_loader)
    trainer.load(args.timestamp)

    quantizer = _resolve_quantizer(trainer)
    original_beta = float(quantizer.prior_beta)

    with_kl = _evaluate(trainer)

    quantizer.prior_beta = 0.0
    without_kl = _evaluate(trainer)
    quantizer.prior_beta = original_beta

    print(_format_result("With KL", with_kl))
    print(_format_result("Without KL", without_kl))
    common_keys = sorted(set(with_kl) & set(without_kl))
    if common_keys:
        print("Differences (Without - With):")
        for key in common_keys:
            print(f"  {key:>10s}: {without_kl[key] - with_kl[key]:.6f}")

    return with_kl, without_kl


if __name__ == "__main__":
    main()
