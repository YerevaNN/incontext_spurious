import torch
import torch.nn as nn

class DinoV2(nn.Module):
    def __init__(self,
                 model,
                 embedding_size: int,
                 transformation_matrix=None,
                 *args, **kwargs):
        super().__init__()

        self.model = torch.hub.load("facebookresearch/dinov2", model=model)

        self.transformation_linear = None
        if transformation_matrix is not None:
            self.transformation_linear = nn.Linear(embedding_size, embedding_size, bias=False)

            transformation_matrix = transformation_matrix.type(self.transformation_linear.weight.data.dtype)
            self.transformation_linear.weight = nn.parameter.Parameter(transformation_matrix)

        self.embedding_size = embedding_size

    def forward(self, x):
        emb = self.model(x)

        if self.transformation_linear is not None:
            emb = self.transformation_linear(emb)

        return emb
