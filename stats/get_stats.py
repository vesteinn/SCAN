import json
import os
import statistics

import matplotlib.pyplot as plt


def read_eval(lines):
    data = []
    for line in lines:
        if line.startswith('{"src": "'):
            try:
                data.append(json.loads(line))
            except:
                breakpoint()
    return data


def read_log(path):
    with open(path) as infile:
        lines = infile.readlines()
        final_stats = lines[-1]
        if "Eval_acc" not in final_stats:
            raise Exception
        sp_line = final_stats.split()
        acc = float(sp_line[4])
        oracle_acc = None
        if "Oracle" in final_stats:
            oracle_acc = float(sp_line[8])
        stats = json.loads(lines[-2])
        preds = read_eval(lines)
    return {"acc": acc, "stats": stats, "oracle_acc": oracle_acc, "preds": preds}


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


def acc(path, key="acc", pattern=""):
    stats = load_folder(path)
    accs = [s[key] for s in stats if pattern in s["path"]]
    try:
        mean = statistics.mean(accs)
    except:
        try:
            mean = accs[0]
        except:
            mean = 0
    try:
        stdev = statistics.stdev(accs)
    except:
        try:
            stdev = accs[0]
        except:
            stdev = 0
    return f"{mean:.1f} ± {stdev:.1f} %", mean, stdev


def exp_1_size_splits(path):
    stats = {}
    for split in [1, 2, 4, 8, 16, 32, 64]:
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
        data = {k: (100 * v[1], 100 * v[2]) for k, v in sorted(data.items())}
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


def search_error(path):
    stats = load_folder(path)
    pred_over_tgt = []
    for log in stats:
        for i in range(len(log["stats"]['tgt_probs'])):
            acc = log["stats"]["accuracy"][i]
            if acc:
                # Only for errors
                continue
            pred_prob = log["stats"]["pred_probs"][i]
            tgt_prob = log["stats"]["tgt_probs"][i]
            pred_over_tgt.append(pred_prob > tgt_prob)
    return sum(pred_over_tgt) / len(pred_over_tgt)


def get_neighbours(model_path, terms):
    tasks = "../data/SCAN/tasks.txt"
    import torch
    import sys
    sys.path.append("../src/scan")
    model = torch.load(model_path)
    model.eval()

    def embed(words):
        encoded = [model.encoder.dictionary[w] for w in words.split()]
        
        hidden = model.init_hidden()
   
        for word in encoded:
            output, hidden = model.encoder(torch.tensor(word).to(model.device()), hidden)

        embedding = hidden[-1].squeeze()
        return embedding

    src_data = []
    src_embs = []
    with open(tasks) as infile:
        for line in infile.readlines():
            src = line.split(" OUT:")[0]
            src = src.split("IN: ")[1]
            src_data.append(src)
            src_embs.append(embed(src))

    top_hits = {}
    for term in terms:
        sim_scores = torch.zeros(len(src_data))
        for i in range(len(src_data)):
            if term == src_data[i]:
                continue

            sim_score = torch.nn.functional.cosine_similarity(embed(term), src_embs[i], dim=0)
            sim_scores[i] = sim_score
        top_hits[term] = sim_scores.topk(5)
    
    results = {}
    for term in top_hits:
        results[term] = [] 
        for idx, val in enumerate(top_hits[term].values):
            text = src_data[top_hits[term].indices[idx]]
            results[term].append((val.item(), text))
    return results


if __name__ == "__main__":
    log_dir = "../logs_38c637/"

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
    exp_2_top_oracle = acc(f"{log_dir}/experiment_2", key="oracle_acc")[0]
    exp_2_ob_oracle = acc(f"{log_dir}/experiment_2/ob", key="oracle_acc")[0]

    exp_2_stats_ob = action_command_stats(f"{log_dir}/experiment_2/")
    exp_2_stats_tp = action_command_stats(f"{log_dir}/experiment_2/ob")
    exp_2_stats = [exp_2_stats_ob, exp_2_stats_tp]

    print("--- Experiment 2 ---")
    print(f"Top performing model: {exp_2_top}")
    print(f"Overal best model: {exp_2_ob}")

    print(f"Top performing model w. length oracle: {exp_2_top_oracle}")
    print(f"Overal best model w. length oracle: {exp_2_ob_oracle}")

    # Action and command length
    plot_exp_2("command_length", "Command Sequence Length", exp_2_stats, "./exp2_command.png", legend=["TP", "OB"])
    plot_exp_2("action_length", "Action Sequence Length", exp_2_stats, "./exp2_action.png", legend=["TP", "OB"])

    # Search error and oracle
    se_tp = search_error(f"{log_dir}/experiment_2")
    se_ob = search_error(f"{log_dir}/experiment_2/ob")

    print(f"Percentage prefer generated (top): {100 * se_tp:.2f} %")
    print(f"Percentage prefer generated (ob): {100 * se_ob:.2f} %")

    #
    # Experiment 3
    #
    exp_3_tl_top = acc(f"{log_dir}/experiment_3/turn_left")[0]
    exp_3_tl_ob = acc(f"{log_dir}/experiment_3/turn_left/ob")[0]
    
    exp_3_j_top = acc(f"{log_dir}/experiment_3/jump")[0]
    exp_3_j_ob = acc(f"{log_dir}/experiment_3/jump/ob")[0]

    print("--- Experiment 3 ---")
    print(f"Top performing model (turn left): {exp_3_tl_top}")
    print(f"Overal best model (turn left): {exp_3_tl_ob}")
    print(f"Top performing model (jump): {exp_3_j_top}")
    print(f"Overal best model (jump): {exp_3_j_ob}")
    
    # Turn left model errors
    for i in range(5):
        
        stats = load_folder(f"{log_dir}/experiment_3/turn_left")[i]
        incorrect = [pred for pred in stats["preds"] if not pred["correct"]]
        tlthrice = [pred for pred in incorrect if "turn left thrice" in pred["src"]]
        tl = [pred for pred in incorrect if "turn left" in pred["src"]]

        print(f"({i}): Number incorrect in turn left for tp: {len(incorrect)}")
        print(f"({i}): Number incorrect containing 'turn left thrice': {len(tlthrice)}")
        print(f"({i}): Number incorrect containing 'turn left': {len(tl)}")

    words = ["run", "jump", "run twice", "jump twice"]
    neighbours = get_neighbours(f"{log_dir}/experiment_3/jump/model_0.pt", words)
    for term in neighbours:
        res = neighbours[term]
        print(f"Neighb for {term}: {res}")
    

