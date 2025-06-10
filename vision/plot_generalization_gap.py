import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_gap(path):
    config_path = os.path.join(path, "configs.json")
    plot_path = os.path.join(path, "plots.npy")
    if not os.path.exists(config_path) or not os.path.exists(plot_path):
        raise FileNotFoundError(f"Missing files in {path}")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    plots = np.load(plot_path, allow_pickle=True).item()
    if "gap" in plots:
        gap = plots["gap"][-1]
    elif "generalization_gap" in plots:
        gap = plots["generalization_gap"][-1]
    else:
        train_key = "mse_train" if "mse_train" in plots else "acc_train"
        test_key = "mse_test" if "mse_test" in plots else "acc_test"
        gap = plots[test_key][-1] - plots[train_key][-1]
    num_enc = cfg["network"].get("num_rb_enc", cfg["network"].get("num_rb"))
    num_dec = cfg["network"].get("num_rb_dec", cfg["network"].get("num_rb"))
    return gap, num_enc, num_dec


def plot_gap(paths, target="encoder", output="gap_plot.png"):
    xs = []
    ys = []
    for p in paths:
        gap, num_enc, num_dec = load_gap(p)
        xs.append(num_enc if target == "encoder" else num_dec)
        ys.append(gap)

    order = np.argsort(xs)
    xs = np.array(xs)[order]
    ys = np.array(ys)[order]

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(f"number of ResNet blocks ({target})")
    plt.ylabel("generalization gap")
    plt.savefig(output)
    print(f"Saved plot to {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot generalization gap")
    parser.add_argument("paths", nargs="+", help="experiment directories")
    parser.add_argument("--target", choices=["encoder", "decoder"], default="encoder")
    parser.add_argument("--output", default="gap_plot.png")
    args = parser.parse_args()

    plot_gap(args.paths, target=args.target, output=args.output)


if __name__ == "__main__":
    main()

