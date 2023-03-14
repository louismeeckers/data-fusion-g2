import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    # model.load_state_dict(checkpoint['optimizer'])