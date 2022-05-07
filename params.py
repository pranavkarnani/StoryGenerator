# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/refined.csv", help="data path")
    parser.add_argument("--dataset_size", type=int, default=50000)
    parser.add_argument("--max_seq_length", type=int, default=600, help="{768, 1024, 1280, 1600}")
    parser.add_argument("--max_generation_length", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--model_name", type=str, default="gpt2", help="gpt2, gpt2-medium, gpt2-large, gpt2-xl")
    parser.add_argument("--sentence_encoder_name", type=str, default="bert-base-uncased")
    parser.add_argument("--decoding_name", type=str, default="nucleus", help="nucleus, sampling, greedy, beam")
    parser.add_argument("--random_seed", type=int, default=73)
    parser.add_argument("--batch_size", type=int, default=8, help="scaler:8")
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument("--gradient_clipping", type=float, default=5.0)

    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--debug", action="store_true")


    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)
    if args.debug is True:
        args.dataset_size = 51
    # Set seeds
    torch.cuda.manual_seed_all(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    return args



def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif optim == 'adamw':
        print("Optimizer: adamw")
        optimizer = torch.optim.AdamW
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


args = parse_args()