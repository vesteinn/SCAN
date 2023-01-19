import argparse
import json
from difflib import SequenceMatcher
import os
import pprint
import random

from collections import defaultdict

import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import generate_scan_dictionary
from data import SCANDataset

from models import GRURNN
from models import LSTMRNN
from transformer_models import SCANTransformer


MODEL_MAP = {
    "lstm": LSTMRNN,
    "gru": GRURNN,
    "transformer": SCANTransformer
}


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
    log_target_probs=False,
    use_oracle=False,
    verbose=False,
):
    model.eval()
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
    total_match = 0
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
            if model.model_type != "transformer":
                output = model(src, tgt, use_oracle=use_oracle, evaluate=True)
            else:
                output = model.predict(src, tgt, dataset.tgt_dict["BOS"], dataset.tgt_dict["EOS"], use_oracle=use_oracle)
            correct_seq = True
            predicted = []
            probs = []
            if len(tgt) != len(output):
                correct_seq = False

            # Remove EOS
            if use_oracle and model.model_type == "transformer":
                tgt = tgt[:-1]
                output = output[:len(tgt)]

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
                    prob = nn.functional.log_softmax(out, dim=-1).squeeze()
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

            accuracy_stats["action_length"][len(tgt) - 1].append(correct_seq)
            accuracy_stats["command_length"][len(src) - 1].append(correct_seq)
            
            if verbose:
                #print(
                #    json.dumps(
                #        {
                #            "src": decoded_src,
                #            "tgt": decoded_tgt,
                #            "pred": decoded_pred,
                #            "correct": correct_seq,
                #            "pred_len": len(decoded_pred.split())
                #        }
                #    )
                #)
                print("---")
                print(f"src: {decoded_src}")
                print(f"tgt: {decoded_tgt}")
                print(f"pred: {decoded_pred}")
                print(f"len: {len(decoded_pred.split())}")
                match = SequenceMatcher(None, decoded_tgt.split(), decoded_pred.split()).find_longest_match(0, len(decoded_tgt.split()), 0, len(decoded_pred.split()))
                total_match += match.size
                print(f"lcs: {match.size}")
            accuracy.append(correct_seq)
            if _ == 3:
                if verbose:
                    avg_lcs = total_match / 3
                    print(f"avg_lcs: {avg_lcs}")
                break

    #assert len(accuracy) == len(dataset)
    accuracy_stats["command_length"] = acc_process(accuracy_stats["command_length"])
    accuracy_stats["action_length"] = acc_process(accuracy_stats["action_length"])
    accuracy_stats["accuracy"] = accuracy
    model.train()
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
    args=None
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    step = 0
    loss_sum = 0
    max_acc = 0
    accs = []

    bos = train_dataset.tgt_dict["BOS"]
    eos = train_dataset.tgt_dict["EOS"]

    accuracy = 0

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
            
            if args.model != "transformer":
                output = model(src, tgt, teacher_forcing=use_teacher_forcing)
            else:
                #bos_tgt = torch.cat((torch.tensor([bos]), tgt))
                #tgt_eos = torch.cat((tgt, torch.tensor([eos])))
                #bos_tgt_eos = torch.cat((bos_tgt, torch.tensor([eos])))

                output = model(src.unsqueeze(dim=0), tgt.unsqueeze(dim=0))
                output = output

                tgt = tgt[1:]
                tgt = torch.cat((tgt, torch.tensor([6]).to(args.device)))
                output = torch.nn.functional.pad(output, (0, len(tgt) - len(output)), value=6)


            # Redundant now that padding has been added - remove
            # works for bsz 1
            pad_pred = torch.zeros(
                max_length - output.shape[0],
                output.shape[-1]
            ).to(device)

            torch.fill(pad_pred, -100)
            output_pad = torch.cat(
                (output.reshape(-1, output.shape[-1]), pad_pred), dim=0
            )

            pred = output.reshape(-1, output.shape[-1]).argmax(dim=-1)
             
            eos_found = 8 in pred
            eos_last = pred[-1] == eos
            cor_len = len(pred) == len(tgt)
            
            if verbose and step % 150 == 0:
                if torch.all(torch.eq(pred, tgt)):
                    accuracy = 1
                print(f"Acc: {accuracy}")
                print(pred)
                print(tgt)
                #if eos_found:
                #    print(f"{eos_found}\t{eos_last}\t{cor_len}")

            tgt_pad = torch.nn.functional.pad(
                tgt, (0, max_length - len(tgt)), value=-100
            )

            # Cross entropy loss over entire sequence
            min_len = max(output.shape[0], tgt.shape[0])
            
            if args.model == "transformer":
                loss = torch.nn.functional.cross_entropy(
                    output.reshape(-1, output.shape[-1]),
                    tgt,
                    #ignore_index=train_dataset.tgt_dict["PAD"]
                )
            else:
                loss = torch.nn.functional.cross_entropy(
                    output_pad,
                    tgt_pad,
                    ignore_index=-100
                )

            #breakpoint()

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
                eval_data, eval_stats = eval(model, eval_dataset, verbose=verbose, use_oracle=use_oracle)
                eval_acc = 100 * sum(eval_data) / len(eval_data)
                accs.append(eval_acc)
                max_acc = max(accs)
                print(
                    f"Step {step} - Eval_acc: {eval_acc:.02f} % over {len(eval_data)} data points (max {max_acc})."
                )
            if step >= steps:
                break
        if step >= steps:
            break
    if name is not None:
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
    
    oracle_string = ""
    if use_oracle:
        oracle_data, oracle_stats = eval(model, eval_dataset, use_oracle=use_oracle)
        oracle_acc = 100 * sum(oracle_data) / len(oracle_data)
        oracle_string = f"(Oracle acc. {oracle_acc:.02f} %) "
        print(f"{oracle_stats}")
    print(f"{json_stats}")
    print(
        f"Step {step} - Eval_acc: {eval_acc:.02f} % {oracle_string}over {len(eval_data)} data points (max {max_acc})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm")
    parser.add_argument("--train", type=str)
    parser.add_argument("--valid", type=str)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=5)
    parser.add_argument("--use_attention", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--use_oracle", action="store_true", default=False)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--log_target_probs", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)

    # Transformer specific
    parser.add_argument("--nheads", type=int, default=6)


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
    
    pad = 0
    if args.model == "transformer":
        pad = 50
    
    train_dataset = SCANDataset(args.train, src_dict, tgt_dict, device=args.device, pad=pad)
    valid_dataset = SCANDataset(args.valid, src_dict, tgt_dict, device=args.device, pad=pad)
    print(f"Loaded train dataset with {len(train_dataset)} entries")
    print(f"Loaded validation dataset with {len(valid_dataset)} entries")

    model = MODEL_MAP[args.model]
    if args.model != "transformer":
        model = model(
            len(src_dict),
            args.hidden_dim,
            args.layers,
            args.dropout,
            src_dict,
            tgt_dict,
            use_attention=args.use_attention,
        )
    else:
        # SCANTransformer
        model = model(
            src_size=len(src_dict),
            tgt_size=len(tgt_dict),
            d_model=args.hidden_dim,
            nhead=args.nheads,
            num_encoder_layers=args.layers,
            num_decoder_layers=args.layers,
            dim_feedforward=args.hidden_dim * 4,
            dropout=args.dropout
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
        args=args
    )
