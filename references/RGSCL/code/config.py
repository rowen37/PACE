import torch
import argparse
import random
import os
import numpy as np


class CONFIG:
    def __init__(self):
        self.args = {}
        self.filepath = '.'  # experiment data path
        self.seed = random.randint(0, 0x7fffffff)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--epoch', type=int, default=200)
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--dim', type=int, default=64, choices=[16, 32, 64, 128])
        parser.add_argument('--lyr', type=int, default=1, choices=[1, 2, 3, 4])
        parser.add_argument('--alpha', type=float, default=.7, choices=[0, .1, .3, .5, .7, .9, 1])
        parser.add_argument('--attn_head', type=int, default=4, choices=[1, 2, 4, 8])
        parser.add_argument('--bsz', type=int, default=32)
        parser.add_argument('--miu', type=float, default=1e-5)
        parser.add_argument('--act_fn', type=str, default='prelu', choices=['relu', 'prelu', 'rrelu'])
        args = parser.parse_args()
        self.args.update(vars(args))
            
    def set_seed(self, seed=42):
        if seed == -1:
            self.seed = random.randint(0, 0x7fffffff)
        else:
            self.seed = seed
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)