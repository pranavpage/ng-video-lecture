import torch
import numpy as np
np.random.seed(42)
torch.manual_seed(42)
# test script for understanding tensor maths
B = 2
T = 3
C = 10
data = np.random.randn(B, T, C)
print(data.shape)
A = torch.tensor(data)
print(A.shape)
print(A)

print("="*20)
print("indexing")
print(A[:, -1, :].shape) # supposed to be last token I guess?
print("="*20)
# yeah, the last token in both sentences

print("="*20)
print("slicing")
print(A[:, -1:, :].shape) #
print("="*20)
# yeah, the last token in both sentences, but as a slice not as an indexed
At = A.transpose(-2,-1)
print(At)
print(At.shape)
print(f"transposed : {At.shape}")

np.random.seed(0)
Astar = torch.tensor(np.random.randn(B,T,C))
print("="*20)
print(f"Astar = {Astar}")
result = torch.cat([A, Astar], dim=2)
print(result.shape)

Areshaped= A.reshape(B*T, C)

print("="*20)
print(Areshaped.shape)
print(Areshaped)
