import torch
import torch.nn as nn
from models.basic import LlamaAttention

class Llama_mntp(nn.modules):
    def __init__(self):
        super().__init__()