import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
np.random.seed(42)

B = 2
T = 3
C = 8
num_heads = 2
x = torch.tensor(np.random.randn(B,T,C), dtype=torch.float32)
print(f"X shape = {x.shape}")
print(f"X = {x}")
print("="*20)

D = C//num_heads # D can be C, C/2 etc depending on num heads of attention
# need to get x (B,T,C) into Q (B, T, D)
query = nn.Linear(C, D, bias=False)
key = nn.Linear(C, D, bias=False)
value = nn.Linear(C, D, bias=False)


Q = query(x) # supposed to be "What am i (tok) looking for?"
K = key(x) # supposed to be "What can be found in me (tok)?"
V = value(x) # supposed to be "what value do I offer" < shaky
print(query)
print(f"Q shape = {Q.shape}")

# let's compute attention
self_att = Q @ K.transpose(-2, -1)
print(self_att.shape)
self_att /= np.sqrt(D) # normalise

mask = torch.tril(torch.ones(T,T))

self_att = self_att.masked_fill(mask == 0, float("-inf"))
print(self_att.shape)
weights = torch.softmax(self_att, dim=-1)
print(weights)
weights = weights @ V
print(f"final output = {weights.shape}")
