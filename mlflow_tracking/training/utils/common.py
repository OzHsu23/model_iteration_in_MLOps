import torch
import json
import time
import random
import numpy as np

def get_device():
    """Return available device: cuda if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_json(path):
    """Load a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Timer:
    """Simple timer class for measuring execution time."""
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer was not started.")
        elapsed = time.time() - self.start_time
        self.start_time = None
        return elapsed
