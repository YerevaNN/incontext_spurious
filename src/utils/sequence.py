import numpy as np
import torch


def unsqueeze_dim(x: torch.Tensor, n_dim: int):
    for _ in range(n_dim - x.dim()):
        x = x.unsqueeze(0)
    return x

def randomly_swap_labels(y):  # For binary classification only
    if np.random.choice([True, False], size=1):
        return -y + 1
    return y

def construct_sequence_from_xy(x: torch.Tensor, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Interleaves the x's and the y's into a single sequence."""
    # x has shape (batch_size, seq_images, emb_size)
    assert x.ndim in [2, 3]
    x = unsqueeze_dim(x, n_dim=3)  # batch_size, seq_images, emb_size 
    
    assert y.ndim in [1, 2]
    y = unsqueeze_dim(y, n_dim=2)  # batch_size, seq_labels
    y = labels[y]  # batch_size, seq_images, emb_size

    batch_size, points, dim = x.shape

    sequence = torch.stack((x, y), dim=2).view(batch_size, 2 * points, dim) # constructing sequence

    return sequence

def construct_inference_sequence(context: torch.Tensor, query: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Interleaves the context(which is tuple of x, y) and query(image-representation) into a single sequence."""
    x, y = context
    # x is (batch_size, seq_len, dim)
    # y is (batch_size, seq_len)

    sequence = construct_sequence_from_xy(x, y, labels)  # batch_size, seq_len, emb_size
    query = unsqueeze_dim(query, n_dim=3)

    if len(sequence) == 1:  # For the same context
        sequence = torch.tile(sequence, dims=(len(query), 1, 1))

    context_query = torch.concat([sequence, query], dim=1)

    return context_query

def sample(arr, num_samples):
    if num_samples == 0:
        return torch.tensor(data=[], dtype=torch.long)

    selected = np.random.choice(arr, num_samples, replace=False)

    return torch.from_numpy(selected)
