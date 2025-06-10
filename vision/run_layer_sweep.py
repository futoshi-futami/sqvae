import argparse
from argparse import Namespace
from plot_generalization_gap import plot_gap
from main import run as run_main


def run_training(cfg_file, gpu, seed, enc_rb=None, dec_rb=None):
    args = Namespace(
        config_file=cfg_file,
        timestamp="",
        save=True,
        dbg=False,
        gpu=gpu,
        seed=seed,
        enc_rb=enc_rb,
        dec_rb=dec_rb,
    )
    return run_main(args)


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
