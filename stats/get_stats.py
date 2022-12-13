import json
import os
import statistics

import matplotlib.pyplot as plt


def read_log(path):
    with open(path) as infile:
        lines = infile.readlines()
        final_stats = lines[-1]
        sp_line = final_stats.split()
        acc = float(sp_line[4])
        stats = json.loads(lines[-2])
        return {"acc": acc, "stats": stats}


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
    height = [d[1] for d in data.values()]
    yerr = [d[2] for d in data.values()]
    x = [f"{p}%" for p in data.keys()]
    xlab = "Percent of commands used for training"
    ylab = "Accuracy on new commands (%)"
    plt.bar(x=x, height=height, yerr=yerr)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if show:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.clf()
    plt.close()


def action_command_stats(path):
    stats = load_folder(path)
    data = {"command_length": {}, "action_length": {}}
 
    for key in data:

        for length in stats[0]["stats"][key]:
            accs = [stat["stats"][key][length] for stat in stats]
            mean = statistics.mean(accs)
            stdev = statistics.stdev(accs)
            data[key][length] = (f"{mean:.1f} ± {stdev:.1f} %", mean, stdev)

    return data


def plot_exp_2(key, xlab, stats, out_file, show=False, legend=[]):
    fig, ax = plt.subplots()
    width = 0.4
    indexes = None
    bars = []
    for idx, stat in enumerate(stats):
        data = stat[key]
        data = {k: (v[1], v[2]) for k, v in sorted(data.items())}
        height = [d[0] for d in data.values()]
        yerr = [d[1] for d in data.values()]
        if indexes is None:
            indexes = data.keys()
        x = [i + idx * width for i in range(len(data.keys()))]
        bars.append(ax.bar(x, height, width, yerr=yerr))

    ylab = "Accuracy on new commands (%)"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xticks([i + width / 2 for i in range(len(indexes))], indexes)

    ax.legend((bars[0][0], bars[1][0]), legend)

    if show:
        plt.show()
    else:
        plt.savefig(out_file)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    log_dir = "../logs"

    #
    # Experiment 1
    # 

    exp_1_top = acc(f"{log_dir}/experiment_1")[0]
    exp_1_ob = acc(f"{log_dir}/experiment_1/ob")[0]
    exp_1_splits = exp_1_size_splits(f"{log_dir}/experiment_1/split_variations/")

    print("--- Experiment 1 ---")
    print(f"Top performing model: {exp_1_top}")
    print(f"Overal best model: {exp_1_ob}")

    for split in exp_1_splits:
        print(f"Split {split}: {exp_1_splits[split][0]}")

    plot_exp_1(exp_1_splits, "./exp_1_size_splits.png")

    # 
    # Experiment 2
    # 

    exp_2_top = acc(f"{log_dir}/experiment_2")[0]
    exp_2_ob = acc(f"{log_dir}/experiment_2/ob")[0]
    exp_2_stats_ob = action_command_stats(f"{log_dir}/experiment_2/")
    exp_2_stats_tp = action_command_stats(f"{log_dir}/experiment_2/ob")
    exp_2_stats = [exp_2_stats_ob, exp_2_stats_tp]

    print("--- Experiment 2 ---")
    print(f"Top performing model: {exp_2_top}")
    print(f"Overal best model: {exp_2_ob}")

    # Action and command length
    plot_exp_2("command_length", "Command Sequence Length", exp_2_stats, "./exp2_command.png", legend=["TP", "OB"])
    plot_exp_2("action_length", "Action Sequence Length", exp_2_stats, "./exp2_action.png", legend=["TP", "OB"])

    # Search error and oracle