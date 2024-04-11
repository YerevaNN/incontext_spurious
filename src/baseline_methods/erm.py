import torch
import torch.nn as nn

from src.baseline_methods.base_method import BaseMethod


class ERM(BaseMethod):
    """Empirical Risk Minimization (ERM) method for binary classification."""
    def __init__(self,
                 n_epochs: int = 100,
                 lr: float = 0.01):
        super(ERM, self).__init__()
        self._n_epochs = n_epochs
        self._lr = lr

    def predict(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            x_test: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        model = nn.Linear(x_train.shape[1], 1)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self._lr)

        # Training loop
        for _ in range(self._n_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

        # Testing
        model.eval()
        with torch.no_grad():
            test_pred = model(x_test)
            test_pred_label = torch.sigmoid(test_pred).squeeze(0)  # Convert to probability

        return test_pred_label
