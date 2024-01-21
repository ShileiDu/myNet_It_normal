import torch
import numpy as np
import random


def str_list(argstr):
    return list(argstr.split(','))




def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
