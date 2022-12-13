import os
import statistics

import matplotlib.pyplot as plt


def read_log(path):
    with open(path) as infile:
        lines = infile.readlines()
        final_stats = lines[-1]
        sp_line = final_stats.split()
        acc = float(sp_line[4])
        return {"acc": acc}


def load_folder(path):
    stats = []
    log_files = [path + "/" + fname for fname in os.listdir(path) if fname[-3:] == "txt"]
    for log in log_files:
        try:
            data = read_log(log)
        except:
            continue
        data["path"] = log
        stats.append(data)
    return stats


def acc(path, pattern=""):
    stats = load_folder(path)
    accs = [s["acc"] for s in stats if pattern in s["path"]]
    mean = statistics.mean(accs)
    stdev = statistics.stdev(accs)
    return f"{mean:.1f} ± {stdev:.1f} %", mean, stdev


def exp_1_size_splits(path):
    stats = {}
    for split in [1, 2, 4, 8, 16, 32]:
        stats[split] = acc(path, pattern=f"_p{split}_")
    return stats


def plot_exp_1(data, out_file, show=False):
    x = range(len(data))
    height = [d[1] for d in data.values()]
    yerr = [d[2] for d in data.values()]
    label = [f"{p}%" for p in data.keys()]
    xlab = "Percent of commands used for training"
    ylab = "Accuracy on new commands (%)"
    plt.bar(x=x, height=height, yerr=yerr, label=label)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if show:
        plt.show()
    else:
        plt.savefig(out_file)

if __name__ == "__main__":
    log_dir = "../logs"

    exp_1_top = acc(f"{log_dir}/experiment_1")[0]
    exp_1_ob = acc(f"{log_dir}/experiment_1/ob")[0]
    exp_1_splits = exp_1_size_splits(f"{log_dir}/experiment_1/split_variations/")

    print("--- Experiment 1 ---")
    print(f"Top performing model: {exp_1_top}")
    print(f"Overal best model: {exp_1_ob}")

    for split in exp_1_splits:
        print(f"Split {split}: {exp_1_splits[split][0]}")

    plot_exp_1(exp_1_splits, "./exp_1_size_splits.png")

    print("--- Experiment 2 ---")