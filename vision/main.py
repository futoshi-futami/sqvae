import os
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
import argparse
import numpy as np  # ensure MKL is initialized before torch
from configs.defaults import get_cfgs_defaults
import torch

from trainer import GaussianSQVAETrainer, VmfSQVAETrainer
from util import set_seeds, get_loader


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
    parser.add_argument(
        "--enc_rb", type=int, default=None,
        help="number of ResNet blocks in encoder")
    parser.add_argument(
        "--dec_rb", type=int, default=None,
        help="number of ResNet blocks in decoder")
    args = parser.parse_args()
    return args


def load_config(args):
    cfgs = get_cfgs_defaults()
    config_path = os.path.join(os.path.dirname(__file__), "configs", args.config_file)
    print(config_path)
    cfgs.merge_from_file(config_path)
    if not hasattr(cfgs.network, "num_rb_enc"):
        cfgs.network.num_rb_enc = cfgs.network.get("num_rb", 2)
    if not hasattr(cfgs.network, "num_rb_dec"):
        cfgs.network.num_rb_dec = cfgs.network.get("num_rb", 2)
    if args.enc_rb is not None:
        cfgs.network.num_rb_enc = args.enc_rb
    if args.dec_rb is not None:
        cfgs.network.num_rb_dec = args.dec_rb
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
        if "mse" in res_test and len(trainer.plots["mse_train"]):
            gap = res_test["mse"] - trainer.plots["mse_train"][-1]
        elif "acc" in res_test and len(trainer.plots["acc_train"]):
            gap = res_test["acc"] - trainer.plots["acc_train"][-1]
        else:
            gap = res_test["loss"] - trainer.plots["loss_train"][-1]
        np.save(os.path.join(trainer.path, "generalization_gap.npy"), gap)

