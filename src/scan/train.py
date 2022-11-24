import argparse
import random
import os

import tqdm

import torch
from torch.utils.data import DataLoader

from data import generate_scan_dictionary
from data import SCANDataset
from models import LSTMRNN
 

MODEL_MAP = {
    "lstm": LSTMRNN
}


def eval(model, dataset, bsz=1):
    accuracy = []
    data_loader = DataLoader(dataset, batch_size=bsz)
    with torch.no_grad():
        for idx, data in tqdm.tqdm(enumerate(data_loader), total=len(dataset)):
            src, tgt = data
            if len(src.shape) > 1 and len(tgt.shape) > 1 \
                    and src.shape[0] == tgt.shape[0] == 1:
                src = src.squeeze()
                tgt = tgt.squeeze()
            if len(src.shape) == 0:
                src = src.unsqueeze(dim=0)
            if len(tgt.shape) == 0:
                tgt = tgt.unsqueeze(dim=0)
            output = model(src, tgt)
            correct_seq = True
            for out_idx, out in enumerate(output):
                if out_idx == 0:
                    # Skip BOS
                    continue
                tgt_idx = out_idx - 1  # ignore BOS
                target = tgt[tgt_idx]
                predicted = torch.nn.functional.softmax(out, dim=-1).squeeze().argmax()
                if target != predicted:
                    # Whole sequence needs to be correct
                    correct_seq = False
            accuracy.append(correct_seq)
    return accuracy  


def train(model, train_dataset, eval_dataset, num_classes, name, lr=0.001, eval_interval=1000, bsz=1, steps=100000, teacher_forcing_ratio=0.5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_loader = DataLoader(train_dataset, batch_size=bsz)
    step = 0
    loss_sum = 0
    max_acc = 0
    accs = []
    for epoch in range(10):
        for idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            src, tgt = data
            if src.shape[0] == tgt.shape[0] == 1:
                src = src.squeeze()
                tgt = tgt.squeeze()
            if len(src.shape) == 0:
                src = src.unsqueeze(dim=0)
                tgt = tgt.unsqueeze(dim=0)

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            output = model(src, tgt, teacher_forcing=use_teacher_forcing)
            loss = torch.tensor(0.0, device=model.device())
            for out_idx, out in enumerate(output):
                if out_idx == 0:
                    # Skip BOS
                    continue
                tgt_idx = out_idx - 1  # ignore BOS
                target_oh  = torch.nn.functional.one_hot(tgt[tgt_idx], num_classes)
                loss += torch.nn.functional.binary_cross_entropy(
                    torch.nn.functional.softmax(out, dim=-1).squeeze(),
                    target_oh.float()
                )

            loss_sum += loss
            if not idx % 100:
                print(f"Step {step} - training loss: {loss_sum / 100}")
                loss_sum = 0

            step += 1
            if loss.requires_grad:
                # Why is this needed...?
                loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            
            if idx and not idx % eval_interval:
                print(f"Step {step} - running eval...")
                eval_data = eval(model, eval_dataset)
                eval_acc = 100 * sum(eval_data) / len(eval_data)
                accs.append(eval_acc)
                max_acc = max(accs)
                print(f"Step {step} - Eval_acc: {eval_acc:.02f} % over {len(eval_data)} data points (max {max_acc}).")
            if step >= steps:
                break
        if step >= steps:
                break
    torch.save(model, name)
    print(f"Finished - running eval...")
    eval_data = eval(model, eval_dataset)
    eval_acc = 100 * sum(eval_data) / len(eval_data)
    accs.append(eval_acc)
    max_acc = max(accs)
    print(f"Step {step} - Eval_acc: {eval_acc:.02f} % over {len(eval_data)} data points (max {max_acc}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--train", type=str)
    parser.add_argument("--valid", type=str)
    parser.add_argument("--name", type=str, default="last_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=5)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    
    args = parser.parse_args()

    print("Starting SCAN training")
    print(f"{args}")
    print(10 * "-")

    cur_path = os.path.dirname(os.path.realpath(__file__))
    tasks = f"{cur_path}/../../data/SCAN/tasks.txt"
    src_dict, tgt_dict = generate_scan_dictionary(tasks, add_bos=True, add_eos=True) 
    train_dataset = SCANDataset(args.train, src_dict, tgt_dict, device=args.device)
    valid_dataset = SCANDataset(args.valid, src_dict, tgt_dict, device=args.device)
   
    print(f"Loaded train dataset with {len(train_dataset)} entries")
    print(f"Loaded validation dataset with {len(valid_dataset)} entries")

    model = MODEL_MAP[args.model]
    # hidden_dim, num_layers, drop_out
    model = model(len(src_dict), args.hidden_dim, args.layers, args.dropout, src_dict, tgt_dict)
    model.to(args.device)
    train(model, train_dataset, valid_dataset, len(tgt_dict), args.name, steps=args.steps, teacher_forcing_ratio=args.teacher_forcing_ratio, eval_interval=args.eval_interval)
    



