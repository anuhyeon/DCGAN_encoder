import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * 3           # y = MulBackward0
z = y + 5           # z = AddBackward0

print(z.grad_fn)  # AddBackward0
for i, fn in enumerate(z.grad_fn.next_functions):
    print(f"{i}: {fn}")