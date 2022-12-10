import argparse
import json
import os
import random

from collections import defaultdict

import tqdm

import torch
from torch.utils.data import DataLoader

from data import generate_scan_dictionary
from data import SCANDataset

from models import GRURNN
from models import LSTMRNN


MODEL_MAP = {"lstm": LSTMRNN, "gru": GRURNN}


def acc_process(acc_dict):
    stats = {}
    for k, v in acc_dict.items():
        acc = sum(v) / len(v)
        stats[k] = acc
    return stats


def eval(
    model,
    dataset,
    bsz=1,
    verbose=False,
    log_target_probs=False,
    use_oracle=False,
    verbose=False,
):

    src_dict = {v: k for k, v in dataset.src_dict.items()}
    tgt_dict = {v: k for k, v in dataset.tgt_dict.items()}

    accuracy_stats = {
        "command_length": defaultdict(list),
        "action_length": defaultdict(list),
        "pred_probs": [],
        "tgt_probs": [],
    }
    accuracy = []
    data_loader = DataLoader(dataset, batch_size=bsz)
    with torch.no_grad():
        for _, data in tqdm.tqdm(enumerate(data_loader), total=len(dataset)):
            src, tgt = data
            if (
                len(src.shape) > 1
                and len(tgt.shape) > 1
                and src.shape[0] == tgt.shape[0] == 1
            ):
                src = src.squeeze()
                tgt = tgt.squeeze()
            if len(src.shape) == 0:
                src = src.unsqueeze(dim=0)
            if len(tgt.shape) == 0:
                tgt = tgt.unsqueeze(dim=0)
            # Safe to add tgt since we are not teacher forcing
            output = model(src, tgt, use_oracle=use_oracle)
            correct_seq = True
            predicted = []
            probs = []
            if len(tgt) != len(output):
                correct_seq = False

            for out_idx, out in enumerate(output):
                prob = torch.nn.functional.log_softmax(out, dim=-1).squeeze()
                pred = prob.argmax()
                probs.append(prob[pred].item())
                predicted.append(pred)

                if len(tgt) == len(output):
                    target = tgt[out_idx]
                    if target != pred:
                        # Whole sequence needs to be correct
                        correct_seq = False

            tgt_probs = []
            if not correct_seq and log_target_probs:
                # Teacher force to get probability of target
                tgt_output = model(src, tgt, teacher_forcing=True)
                for out_idx, out in enumerate(tgt_output):
                    prob = torch.nn.functional.log_softmax(out, dim=-1).squeeze()
                    tgt_prob = prob[tgt[out_idx]]
                    tgt_probs.append(tgt_prob.item())
            else:
                tgt_probs = None
            accuracy_stats["pred_probs"].append(sum(probs))
            if tgt_probs is not None:
                tgt_probs = sum(tgt_probs)
            accuracy_stats["tgt_probs"].append(tgt_probs)

            decoded_src = " ".join([src_dict[t.item()] for t in src])
            decoded_tgt = " ".join([tgt_dict[t.item()] for t in tgt])
            decoded_pred = " ".join([tgt_dict[t.item()] for t in predicted])
            try:
                assert (decoded_tgt == decoded_pred) == correct_seq
            except:
                breakpoint()

            # neg 1 since eos
            accuracy_stats["action_length"][len(tgt) - 1].append(correct_seq)
            accuracy_stats["command_length"][len(src) - 1].append(correct_seq)
            if verbose:
                print(
                    json.dumps(
                        {
                            "src": decoded_src,
                            "tgt": decoded_tgt,
                            "pred": decoded_pred,
                            "correct": correct_seq,
                        }
                    )
                )

            accuracy.append(correct_seq)
    assert len(accuracy) == len(dataset)
    accuracy_stats["command_length"] = acc_process(accuracy_stats["command_length"])
    accuracy_stats["action_length"] = acc_process(accuracy_stats["action_length"])
    accuracy_stats["accuracy"] = accuracy
    return accuracy, accuracy_stats


max_length = 64


def train(
    model,
    train_dataset,
    eval_dataset,
    name,
    lr=0.001,
    log_interval=1000,
    eval_interval=1000,
    bsz=1,
    steps=100000,
    teacher_forcing_ratio=0.5,
    use_oracle=False,
    device="cpu",
    log_target_probs=False,
    verbose=False,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    step = 0
    loss_sum = 0
    max_acc = 0
    accs = []
    for _epoch in range(1 + steps // len(data_loader)):
        for idx, data in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            optimizer.zero_grad()
            src, tgt = data
            if src.shape[0] == tgt.shape[0] == 1:
                src = src.squeeze()
                tgt = tgt.squeeze()
            if len(src.shape) == 0:
                src = src.unsqueeze(dim=0)
                tgt = tgt.unsqueeze(dim=0)

            use_teacher_forcing = (
                True if random.random() < teacher_forcing_ratio else False
            )
            output = model(src, tgt, teacher_forcing=use_teacher_forcing)

            # works for bsz 1
            pad_zeros = torch.zeros(max_length - output.shape[0], output.shape[-1]).to(
                device
            )
            torch.fill(pad_zeros, -100)
            output_pad = torch.cat(
                (output.reshape(-1, output.shape[-1]), pad_zeros), dim=0
            )

            tgt_pad = torch.nn.functional.pad(
                tgt, (0, max_length - len(tgt)), value=-100
            )
            loss = torch.nn.functional.cross_entropy(output_pad, tgt_pad)
            loss.backward()

            loss_sum += loss
            if not step % log_interval:
                print(f"Step {step} - training loss: {loss_sum / log_interval}")
                loss_sum = 0

            step += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            if not step % eval_interval:
                print(f"Step {step} - running eval...")
                eval_data, eval_stats = eval(model, eval_dataset)
                eval_acc = 100 * sum(eval_data) / len(eval_data)
                accs.append(eval_acc)
                max_acc = max(accs)
                oracle_string = ""
                # For debugging - too much to run all the time..
                # if use_oracle:
                #    oracle_data, oracle_stats = eval(model, eval_dataset, use_oracle=use_oracle)
                #    oracle_acc = 100 * sum(oracle_data) / len(oracle_data)
                #    oracle_string = f"(Oracle acc. {oracle_acc:.02f} %) "
                print(
                    f"Step {step} - Eval_acc: {eval_acc:.02f} % {oracle_string}over {len(eval_data)} data points (max {max_acc})."
                )
            if step >= steps:
                break
        if step >= steps:
            break
    torch.save(model, name)
    print(f"Finished - running eval...")
    eval_data, eval_stats = eval(
        model, eval_dataset, log_target_probs=log_target_probs, verbose=verbose
    )
    eval_acc = 100 * sum(eval_data) / len(eval_data)
    accs.append(eval_acc)
    max_acc = max(accs)
    try:
        json_stats = json.dumps(eval_stats)
    except:
        breakpoint()

    print(f"{json_stats}")
    if use_oracle:
        oracle_data, oracle_stats = eval(model, eval_dataset, use_oracle=use_oracle)
        oracle_acc = 100 * sum(oracle_data) / len(oracle_data)
        oracle_string = f"(Oracle acc. {oracle_acc:.02f} %) "
    print(
        f"Step {step} - Eval_acc: {eval_acc:.02f} % {oracle_string}over {len(eval_data)} data points (max {max_acc})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm")
    parser.add_argument("--train", type=str)
    parser.add_argument("--valid", type=str)
    parser.add_argument("--name", type=str, default="last_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=5)
    parser.add_argument("--use_attention", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--use_oracle", action="store_true", default=False)
    parser.add_argument("--use_concat_hidden", action="store_true", default=False)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--log_target_probs", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print("Starting SCAN training")
    print(f"{args}")
    print(10 * "-")

    cur_path = os.path.dirname(os.path.realpath(__file__))
    data_path = f"{cur_path}/../../data/SCAN"
    tasks = f"{data_path}/tasks.txt"
    src_dict, tgt_dict = generate_scan_dictionary(tasks, add_bos=True, add_eos=True)
    train_dataset = SCANDataset(args.train, src_dict, tgt_dict, device=args.device)
    valid_dataset = SCANDataset(args.valid, src_dict, tgt_dict, device=args.device)
    print(f"Loaded train dataset with {len(train_dataset)} entries")
    print(f"Loaded validation dataset with {len(valid_dataset)} entries")

    model = MODEL_MAP[args.model]
    # hidden_dim, num_layers, drop_out
    model = model(
        len(src_dict),
        args.hidden_dim,
        args.layers,
        args.dropout,
        src_dict,
        tgt_dict,
        use_attention=args.use_attention,
        use_concat_hidden=args.use_concat_hidden,
    )
    model.to(args.device)
    train(
        model,
        train_dataset,
        valid_dataset,
        args.name,
        steps=args.steps,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        eval_interval=args.eval_interval,
        use_oracle=args.use_oracle,
        device=args.device,
        log_target_probs=args.log_target_probs,
        verbose=args.verbose,
    )
