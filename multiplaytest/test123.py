import torch
import numpy as np

in_tensor = torch.rand((16, 32, 2, 3, 128, 72))

_num = in_tensor.shape[2]

print("[np.arange(_num), np.arange(_num)[::-1]]", [np.arange(_num), np.arange(_num)[::-1]])
# First we duplicate each batch entry and swap player axes when relevant
in_tensor = in_tensor[
            :,
            :,
            np.array([np.arange(_num), np.arange(_num)[::-1]]),
            ...
            ]

#print(in_tensor.shape)
print("duplicate and swap", in_tensor.shape)

print("transpose", in_tensor.transpose(1, 2).shape)

# Then we swap the new dims and channel dims so we can combine them with the batch dims
in_tensor = torch.flatten(
    in_tensor.transpose(1, 2),
    start_dim=0,
    end_dim=1
)

print("combine batch", in_tensor.shape)

# Finally, combine channel and player dims
out = torch.flatten(in_tensor, start_dim=1, end_dim=2)

print(out.shape)


_fatten = torch.tensor([[[1, 2], [3, 4]],  [[5, 6], [7, 8]]])
print(_fatten)
print(torch.flatten(_fatten, start_dim=1))
print(torch.flatten(_fatten, start_dim=0, end_dim=1))