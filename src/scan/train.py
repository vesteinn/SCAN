import argparse
import random
import os
import numpy as np
import tqdm

import torch
from torch.utils.data import DataLoader

from data import generate_scan_dictionary
from data import SCANDataset

from models_v import GRURNN
from models_v import LSTMRNN
import pdb
 

MODEL_MAP = {
    "lstm": LSTMRNN,
    "gru": GRURNN
}


def eval(model, dataset, bsz=1):
    accuracy = []
    data_loader = DataLoader(dataset, batch_size=bsz)
    error = []
    accuracy_greedy=[]
    error_sum = 0
    greedy = 0
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
            greedy_correct_seq = False
            for out_idx, out in enumerate(output):
#                if out_idx == 0:
#                    # Skip BOS
#                    continue
                tgt_idx = out_idx  # ignore BOS
                target = tgt[tgt_idx]
                # output is logsoftmax
                predicted = torch.exp(out).argmax() #torch.nn.functional.softmax(out, dim=-1).squeeze().argmax()
                gre, _ = torch.exp(out).topk(2)
                #print(gre)
                if gre[0][0] == gre[0][1]:
                    greedy += 1
                    if target == predicted:
                        error_sum += 1
                    greedy_correct_seq = True
        
                
                if target != predicted:
                    # Whole sequence needs to be correct
                    correct_seq = False
                    error_sum += 1

            error.append(correct_seq == False or greedy_correct_seq == True)
            accuracy.append(correct_seq)
            accuracy_greedy.append(greedy_correct_seq)
    return accuracy, greedy/error_sum, sum(accuracy_greedy)/sum(error)


def train(train_dataset, eval_dataset,MODEL_MAP, model_name,hidden_dim, layers, dropout, src_dict, tgt_dict, use_attention,device,num_classes, name, lr=0.001, eval_interval=1000, bsz=1, steps=100000, teacher_forcing_ratio=0.5):

    # step = 0
    max_acc = 0
    
    #to append evaluation for each run
    accs = []
    best_eval_acc = 0
    for epoch in range(1):
        model_new = MODEL_MAP[model_name]
        model = model_new(len(src_dict), hidden_dim, layers, dropout, src_dict, tgt_dict, use_attention)
        model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # model = model_init
        # print(model)
        # pdb.set_trace()
        # random_list = np.random.choice([j for j in range(len(train_dataset))],steps)
        # data_ = [train_dataset[i] for i in random_list]
        # data_loader = DataLoader(data_, batch_size=bsz)
        
        #"step" is only for print every 1000 step
        acc_step = []
        loss_sum = 0
        loss_step = 0
        for i in range(steps):
        # for idx, data in enumerate(data_loader):
            index = np.random.randint(len(train_dataset))
            optimizer.zero_grad()
            src, tgt = train_dataset[index]
            if src.shape[0] == tgt.shape[0] == 1:
                src = src.squeeze()
                tgt = tgt.squeeze()
            if len(src.shape) == 0:
                src = src.unsqueeze(dim=0)
                tgt = tgt.unsqueeze(dim=0)

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            output = model(src, tgt, teacher_forcing=use_teacher_forcing)
            loss = torch.tensor(0.0, device=model.device())
            correct_seq = True
            
            for out_idx, out in enumerate(output):
#                if out_idx == 0:
#                    # Skip BOS
#                    continue
#                tgt_idx = out_idx - 1  # ignore BOS
#                target = tgt[tgt_idx]

                tgt_idx = out_idx
                target = tgt[tgt_idx]
                predicted = torch.exp(out).argmax() #torch.nn.functional.softmax(out, dim=-1).squeeze().argmax()
#                if out_idx == 0:
#                    print(tgt[tgt_idx])
#                    print(predicted)
                if target != predicted:
                    # Whole sequence needs to be correct
                    correct_seq = False
                    
                target_oh  = torch.nn.functional.one_hot(tgt[tgt_idx], num_classes)
                loss += torch.nn.functional.binary_cross_entropy(
                    torch.nn.functional.softmax(out, dim=-1).squeeze(),
                    target_oh.float()
                )
            acc_step.append(correct_seq)

            loss_sum += loss
            loss_step += loss
            
            if not i % 1000:
                print(f"Epoch {epoch+1} Step {i} - training loss: {loss_step / 1000} - training accuracy: {sum(acc_step)/10:.02f} %")
                loss_step = 0
                acc_step = []

            # step += 1
            if loss.requires_grad:
                # Why is this needed...?
                loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            
        #print valid result
        print("Running eval...")
        eval_data, greedy_com, greedy_seq= eval(model, eval_dataset)
        eval_acc = 100 * sum(eval_data) / len(eval_data)
        accs.append(eval_acc)
        max_acc = max(accs)
        
        
        
        #print train result
#        train_data,_ = eval(model, train_dataset)
#        train_acc = 100 * sum(train_data) / len(train_data)
#        
#        print(f"Epoch {epoch+1} - Train_acc: {train_acc:.02f} - Eval_acc: {eval_acc:.02f} % over {len(eval_data)} data points (max {max_acc}).")
        print(f"Epoch {epoch+1} - Eval_acc: {eval_acc:.02f} % over {len(eval_data)} data points (max {max_acc}).")
        print(f"- Greedy/Com: {greedy_com}, -Greedy/Seq: {greedy_seq}")
        if eval_acc >= best_eval_acc:
            best_eval_acc = eval_acc
            print("Found new best model on dev set!")
            # torch.save(model.state_dict(), 'model_best.std')
        # if step >= steps:
        #     break

    # model.load_state_dict(torch.load('model_best.std'))
    print(f"Finished - running eval...")
#    eval_data, greedy_com, greedy_seq= eval(model, eval_dataset)
#    eval_acc = 100 * sum(eval_data) / len(eval_data)
#    max_acc = max(accs)
    #average over 10 runs
    avg_eval_acc = np.mean(accs)
    print(f"Final - AVG_eval_acc: {avg_eval_acc:.02f}%.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default = 'gru')
    parser.add_argument("--train", type=str)
    parser.add_argument("--valid", type=str)
    parser.add_argument("--name", type=str, default="last_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=5)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--use_attention", type=str, default=True)
    
    args = parser.parse_args()

    print("Starting SCAN training")
    print(f"{args}")
    print(10 * "-")

    cur_path = os.path.dirname(os.path.realpath(__file__))
    data_path = f"{cur_path}/data"
    tasks = f"{data_path}/tasks.txt"
    args.train = f"{data_path}/tasks_train_addprim_turn_left.txt"
    args.valid = f"{data_path}/tasks_test_addprim_turn_left.txt"
    src_dict, tgt_dict = generate_scan_dictionary(tasks, add_bos=True, add_eos=True) 
    
    #print(tgt_dict)

    train_dataset = SCANDataset(args.train, src_dict, tgt_dict, device=args.device)
    valid_dataset = SCANDataset(args.valid, src_dict, tgt_dict, device=args.device)
    

    
   
    print(f"Loaded train dataset with {len(train_dataset)} entries")
    print(f"Loaded validation dataset with {len(valid_dataset)} entries")

    # model = MODEL_MAP[args.model]
    # # hidden_dim, num_layers, drop_out
    # model = model(len(src_dict), args.hidden_dim, args.layers, args.dropout, src_dict, tgt_dict, args.use_attention)
    # model.to(args.device)
    # pdb.set_trace()
    train(train_dataset= train_dataset, eval_dataset = valid_dataset, MODEL_MAP = MODEL_MAP,model_name = args.model, hidden_dim=args.hidden_dim, layers=args.layers, dropout=args.dropout, src_dict = src_dict, tgt_dict = tgt_dict, use_attention=args.use_attention, device = args.device,num_classes=len(tgt_dict), name = args.name, steps=args.steps, teacher_forcing_ratio=args.teacher_forcing_ratio, eval_interval=args.eval_interval)
    
    

