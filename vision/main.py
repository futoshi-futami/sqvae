import os
import argparse
from typing import Dict, Optional, Sequence

import torch

from configs.defaults import get_cfgs_defaults
from trainer import GaussianSQVAETrainer, VmfSQVAETrainer
from util import set_seeds, get_loader


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the ArgumentParser used by both the CLI and library callers."""

    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument("-c", "--config_file", default="", help="config file")
    parser.add_argument(
        "-ts",
        "--timestamp",
        default="",
        help="saved path (random seed + date)",
    )
    parser.add_argument("--save", action="store_true", help="save trained model")
    parser.add_argument(
        "--dbg", action="store_true", help="print losses per epoch"
    )
    parser.add_argument("--gpu", default="0", help="index of gpu to be used")
    parser.add_argument(
        "--seed", type=int, default=0, help="seed number for randomness"
    )
    parser.add_argument(
        "--cdvib_beta",
        type=float,
        default=None,
        help="coefficient for the CDVIB-style KL regularizer",
    )
    return parser


def arg_parse(argv: Optional[Sequence[str]] = None):
    parser = build_arg_parser()
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(list(argv))


def load_config(args):
    cfgs = get_cfgs_defaults()
    config_path = os.path.join(os.path.dirname(__file__), "configs", args.config_file)
    print(config_path)
    cfgs.merge_from_file(config_path)
    cfgs.train.seed = args.seed
    cfgs.flags.save = args.save
    cfgs.flags.noprint = not args.dbg
    cfgs.path_data = cfgs.path
    cfgs.path = os.path.join(cfgs.path, cfgs.path_specific)
    if args.cdvib_beta is not None:
        cfgs.quantization.cdvib_beta = args.cdvib_beta
    if cfgs.model.name.lower() == "vmfsqvae":
        cfgs.quantization.dim_dict += 1
    cfgs.flags.var_q = not(cfgs.model.param_var_q=="gaussian_1" or
                                        cfgs.model.param_var_q=="vmf")
    cfgs.freeze()
    flgs = cfgs.flags
    return cfgs, flgs


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    if args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfgs, flgs = load_config(args)
    print("[Checkpoint path] "+cfgs.path)
    print(cfgs)

    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.seed)

    train_loader, val_loader, test_loader = get_loader(
        cfgs.dataset.name, cfgs.path_dataset, cfgs.train.bs, cfgs.nworker)
    print("Complete dataload")

    print("=== {} ===".format(cfgs.model.name.upper()))
    if cfgs.model.name == "GaussianSQVAE":
        trainer = GaussianSQVAETrainer(cfgs, flgs, train_loader, val_loader, test_loader)
    elif cfgs.model.name == "VmfSQVAE":
        trainer = VmfSQVAETrainer(cfgs, flgs, train_loader, val_loader, test_loader)
    else:
        raise Exception("Undefined model.")

    if args.timestamp == "":
        trainer.main_loop()

    if flgs.save:
        trainer.load(args.timestamp)
        print("Best models were loaded!!")
        test_result = trainer.test()
    else:
        test_result = trainer.test()

    return {
        "test_result": test_result,
        "checkpoint_dir": trainer.path,
    }


if __name__ == "__main__":
    print("main.py")
    parsed_args = arg_parse()
    run_experiment(parsed_args)

