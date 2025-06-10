import os
import argparse
import subprocess
import glob
import json

from configs.defaults import get_cfgs_defaults
from plot_generalization_gap import plot_gap


def get_config(cfg_file):
    cfgs = get_cfgs_defaults()
    config_path = os.path.join(os.path.dirname(__file__), "configs", cfg_file)
    cfgs.merge_from_file(config_path)
    base_path = os.path.join(cfgs.path, cfgs.path_specific)
    network_name = cfgs.network.name
    # Older configuration files might not specify a training seed.  Provide
    # a default to avoid AttributeError when accessing the field.
    seed = getattr(cfgs.train, "seed", None)
    return base_path, network_name, seed, config_path


def run_training(cfg_file, gpu, seed, enc_rb=None, dec_rb=None):
    base_path, network_name, _, _ = get_config(cfg_file)
    pattern = os.path.join(base_path, f"{network_name}_seed{seed}_*")
    before = set(glob.glob(pattern))

    cmd = ["python", os.path.join(os.path.dirname(__file__), "main.py"),
           "-c", cfg_file, "--save", "--seed", str(seed)]
    if gpu:
        cmd.extend(["--gpu", gpu])
    if enc_rb is not None:
        cmd.extend(["--enc_rb", str(enc_rb)])
    if dec_rb is not None:
        cmd.extend(["--dec_rb", str(dec_rb)])

    env = os.environ.copy()
    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    subprocess.run(cmd, check=True, env=env)

    after = set(glob.glob(pattern))
    new_dirs = list(after - before)
    if not new_dirs:
        raise RuntimeError("Experiment directory not found after training")
    return new_dirs[0]


def sweep(cfg_file, gpu, seed, values, target):
    paths = []
    for rb in values:
        if target == "encoder":
            p = run_training(cfg_file, gpu, seed, enc_rb=rb)
        else:
            p = run_training(cfg_file, gpu, seed, dec_rb=rb)
        paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Run layer sweep and plot gaps")
    parser.add_argument("-c", "--config", required=True, help="config file")
    parser.add_argument("--gpu", default="0", help="GPU index")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--values", nargs="*", type=int, default=[2,4,6,8],
                        help="ResNet layer counts")
    args = parser.parse_args()

    enc_paths = sweep(args.config, args.gpu, args.seed, args.values, "encoder")
    dec_paths = sweep(args.config, args.gpu, args.seed, args.values, "decoder")

    plot_gap(enc_paths, target="encoder", output="gap_encoder.png")
    plot_gap(dec_paths, target="decoder", output="gap_decoder.png")


if __name__ == "__main__":
    main()
