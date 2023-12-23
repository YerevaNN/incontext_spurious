import torch.nn as nn

from torchvision import models


class ResNet50(nn.Module):
    def __init__(self,
                 pretrained: bool,
                 embedding_size: int,
                 transformation_matrix=None,
                 *args, **kwargs):
        super().__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        self.without_linear = nn.Sequential(*list(self.model.children())[:-1])

        self.transformation_linear = None
        if transformation_matrix is not None:
            self.transformation_linear = nn.Linear(embedding_size, embedding_size, bias=False)

            transformation_matrix = transformation_matrix.type(self.transformation_linear.weight.data.dtype)
            self.transformation_linear.weight = nn.parameter.Parameter(transformation_matrix)

        self.embedding_size = embedding_size

    def forward(self, x):
        emb = self.without_linear(x).squeeze((2, 3))

        if self.transformation_linear is not None:
            emb = self.transformation_linear(emb)

        return emb