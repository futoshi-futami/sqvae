import os
import argparse
import numbers
from configs.defaults import get_cfgs_defaults
import torch

from trainer import GaussianSQVAETrainer, VmfSQVAETrainer
from util import set_seeds, get_loader


def _coerce_metric_value(value):
    if isinstance(value, numbers.Number):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (ValueError, TypeError):
            return None
    return None


def _print_metric_table(metrics_with, metrics_without, beta_with):
    metric_names = sorted(set(metrics_with.keys()) & set(metrics_without.keys()))
    header = (
        f"Metric comparison (beta={beta_with:g} vs beta=0):" if beta_with is not None
        else "Metric comparison (configured beta vs beta=0):"
    )
    print("\n" + header)
    print("=" * len(header))
    print(f"{'metric':<16}{'beta':>12}{'no_kl':>12}{'delta':>12}")
    for name in metric_names:
        val_with = _coerce_metric_value(metrics_with[name])
        val_without = _coerce_metric_value(metrics_without[name])
        if val_with is None or val_without is None:
            continue
        delta = val_without - val_with
        print(f"{name:<16}{val_with:>12.6f}{val_without:>12.6f}{delta:>12.6f}")


def arg_parse():
    parser = argparse.ArgumentParser(
            description="main.py")
    parser.add_argument(
        "-c", "--config_file", default="", help="config file")
    parser.add_argument(
        "-ts", "--timestamp", default="", help="saved path (random seed + date)")
    parser.add_argument(
        "--save", action="store_true", help="save trained model")
    parser.add_argument(
        "--dbg", action="store_true", help="print losses per epoch")
    parser.add_argument(
        "--gpu", default="0", help="index of gpu to be used")
    parser.add_argument(
        "--seed", type=int, default=0, help="seed number for randomness")
    args = parser.parse_args()
    return args


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
    if cfgs.model.name.lower() == "vmfsqvae":
        cfgs.quantization.dim_dict += 1
    cfgs.flags.var_q = not(cfgs.model.param_var_q=="gaussian_1" or
                                        cfgs.model.param_var_q=="vmf")
    cfgs.freeze()
    flgs = cfgs.flags
    return cfgs, flgs


if __name__ == "__main__":
    print("main.py")
    
    ## Experimental setup
    args = arg_parse()
    if args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfgs, flgs = load_config(args)
    print("[Checkpoint path] "+cfgs.path)
    print(cfgs)
    
    ## Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.seed)

    ## Data loader
    train_loader, val_loader, test_loader = get_loader(
        cfgs.dataset.name, cfgs.path_dataset, cfgs.train.bs, cfgs.nworker)
    print("Complete dataload")

    ## Trainer
    print("=== {} ===".format(cfgs.model.name.upper()))
    if cfgs.model.name == "GaussianSQVAE":
        trainer = GaussianSQVAETrainer(cfgs, flgs, train_loader, val_loader, test_loader)
    elif cfgs.model.name == "VmfSQVAE":
        trainer = VmfSQVAETrainer(cfgs, flgs, train_loader, val_loader, test_loader)
    else:
        raise Exception("Undefined model.")

    ## Main
    if args.timestamp == "":
        trainer.main_loop()
    if flgs.save:
        trainer.load(args.timestamp)
        print("Best models were loaded!!")
        res_test = trainer.test()
        model_module = getattr(trainer.model, "module", trainer.model)
        quantizer = getattr(model_module, "quantizer", None)
        if quantizer is not None and hasattr(quantizer, "set_prior_beta"):
            original_beta = getattr(quantizer, "prior_beta", 0.0)
            beta_value = float(original_beta) if original_beta is not None else None
            quantizer.set_prior_beta(0.0)
            try:
                res_test_no_kl = trainer.evaluate_once("test")
            finally:
                quantizer.set_prior_beta(original_beta)
            _print_metric_table(res_test, res_test_no_kl, beta_value)

