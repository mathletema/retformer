import torch
from torch.utils import data
from torch import nn, optim
from torch.nn import functional as F

import numpy as np

from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")