from cs336_basics.losses import (data_loading, cross_entropy, AdamW,
                    gradient_clipping,learning_rate_schedule,
                    save_checkpoint)
from cs336_basics.model import TransformerLM
import torch
from collections.abc import Callable
from typing import Optional
from collections.abc import Iterable
import numpy as np
import csv
import argparse
from time import perf_counter
import os
import wandb

def main():
    #Choose hyperparameters
    vocab_size = 10_000
    context_length = 256
    d_model = 512
    d_ff = None #for "swiglu" d_ff = 8/3*d_m for "silu" d_ff = 4*d_m
    theta = 10_000
    num_layers = 4
    num_heads = 16
    betas = (0.9, 0.95)


    #input through terminal
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str,default="cuda:0")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--eval_batches", type=int, default=5)
    parser.add_argument(
        "--train_path",
        type=str,
        default="artifacts/tokenized/tinystories_train.bin",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="artifacts/tokenized/tinystories_valid.bin",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="artifacts/checkpoints/last.pt",
    )
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--no_pos_emb", action="store_true")
    parser.add_argument("--ffn_type", type=str,default="swiglu")
    parser.add_argument("--norm_position", type=str, default="pre")

    args = parser.parse_args()

    num_steps = args.num_steps
    a_max = args.lr
    a_min = args.min_lr
    T_w = args.warmup_steps
    T_c = num_steps - 1
    weight_decay = args.weight_decay
    M = args.max_grad_norm
    milestone_num = args.eval_interval
    eval_batches = args.eval_batches
    lr = a_max

    if args.ffn_type == "swiglu":
        d_ff = 1344
    elif args.ffn_type == "silu":
        d_ff = 4 * d_model

    #Read Data -> X
    token_dtype = np.uint16
    train_data = np.memmap(args.train_path,
                        dtype=token_dtype,
                        mode="r")
    val_data = np.memmap(args.val_path,
                        dtype=token_dtype,
                        mode="r")
    #TransformerLM
    model_obj = TransformerLM(vocab_size,context_length,
                                num_layers,d_model,num_heads,
                                d_ff,theta,args.device,
    no_norm=args.no_norm,no_pos_emb=args.no_pos_emb,ffn_type=args.ffn_type,
    norm_position=args.norm_position)

    #Optmizer
    optimizer = AdamW(model_obj.parameters(),
                    betas,weight_decay,lr)

    run_num = 1
    #"dynamic" csv
    log_path = f"artifacts/checkpoints/run_{run_num}.csv"
    while os.path.exists(log_path):
        run_num +=1
        log_path = f"artifacts/checkpoints/run_{run_num}.csv"

    #WANDB configuration
    run = wandb.init(project="cs336-tinystories",
                    name=f"run_{run_num}",
                    config=vars(args))

    start_time = perf_counter() #start timer

    with open(log_path,"x",
            newline="", encoding="utf-8") as f: #save csv file
        writer = csv.writer(f)
        writer.writerow(["step", "elapsed_time", "type", "loss", "lr"])
        for i in range(num_steps):
            optimizer.zero_grad()
            X_train,Y_train = data_loading(train_data,
                                        args.batch_size,
                                        context_length,
                                        args.device)

            #logits
            logits_train = model_obj(X_train)

            loss = cross_entropy(logits_train,Y_train)
            loss.backward()
            #grad clipping + save for wandb.ai
            grad_norm = gradient_clipping(model_obj.parameters(),M).item()
            clip_scale = (1.0 if grad_norm <= M
                        else M/(grad_norm+1e-6))

            lr = learning_rate_schedule(i,a_max,a_min,
                                        T_w,T_c)
            for group in optimizer.param_groups:
                    group["lr"] = lr
            optimizer.step()

            #upload results
            print(f"step: {i+1} | lr: {lr:.2e} | loss: {loss.item():.4f}")
            run.log(
                    {"train_loss": loss.item(),
                    "lr": lr,
                    "elapsed_time": perf_counter() - start_time,
                    "grad_norm_before_clipping": grad_norm,
                    "clip_scale": clip_scale}, step=i+1)
            writer.writerow([i+1, perf_counter()-start_time,
                            "train", loss.item(), lr])
            f.flush()

            if (i+1) % milestone_num == 0 or (i+1) == num_steps:
                total_loss_val = torch.zeros_like(loss)
                for j in range(eval_batches):
                    X_val, Y_val = data_loading(val_data,
                                            args.batch_size,
                                            context_length,
                                            args.device)
                    with torch.no_grad():
                                logits_val = model_obj(X_val)
                                total_loss_val += cross_entropy(logits_val,Y_val)
                save_checkpoint(model_obj,optimizer,
                                            i+1,args.checkpoint_path)
                avg_loss = (total_loss_val/eval_batches).item()
                print(f"Validation loss: {avg_loss:.4f}")
                run.log({
                    "val_loss:": avg_loss}, step = i+1
                )
                writer.writerow([i+1, perf_counter()-start_time,
                                    "val", avg_loss,lr])
                f.flush()
    run.finish()

if __name__ == "__main__":
      main()