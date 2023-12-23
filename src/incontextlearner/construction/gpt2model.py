import torch
import torch.nn as nn
import torchmetrics
import lightning as L
from lightning.pytorch.utilities import grad_norm

from omegaconf import DictConfig
from hydra.utils import instantiate

from src.utils import WorstGroupAccuracy, WorstGroupLoss
from src.utils import construct_sequence, combine, randomly_swap_labels

from transformers import GPT2Config, GPT2Model

class InContextLearner(L.LightningModule):
    def __init__(
            self,
            n_dims: int,
            icl_params: DictConfig,
            optimizer_conf: dict,
            num_classes: int,
            num_confounders: int,
            label_emb_mode: str = "opposite",
            label_ratios: list = [],
            minority_ratios: list = [],
            *args, **kwargs
        ):
        super().__init__()
        self.save_hyperparameters()
        
        self._n_dims = n_dims
        self._configuration = GPT2Config(**icl_params)

        self._optimizer_conf = optimizer_conf

        self._num_classes = num_classes
        self._num_confounders = num_confounders
        self._num_groups = num_classes * num_confounders

        self._label_emb_mode = label_emb_mode
        self.labels = self._create_labels_emb()

        self.embedding = nn.Linear(self._n_dims, icl_params['n_embd'])
        self.backbone = GPT2Model(self._configuration)
        self.out = nn.Linear(icl_params['n_embd'], num_classes)

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.accuracy = dict()
        self.accuracy_wg = dict()
        self.loss = dict()
        self.loss_wg = dict()

        for set_name in ['train', 'val', 'test']:
            self.accuracy[set_name] = torchmetrics.Accuracy(task="multiclass", num_classes=self._num_classes)
            self.accuracy_wg[set_name] = WorstGroupAccuracy(num_groups=self._num_groups)
            self.loss[set_name] = torchmetrics.MeanMetric()
            self.loss_wg[set_name] = WorstGroupLoss(num_groups=self._num_groups)

            setattr(self, f"{set_name}_accuracy", self.accuracy[set_name])
            setattr(self, f"{set_name}_accuracy_wg", self.accuracy_wg[set_name])
            setattr(self, f"{set_name}_loss", self.loss[set_name])
            setattr(self, f"{set_name}_loss_wg", self.loss_wg[set_name])

        self._label_ratios = label_ratios
        self._minority_ratios = minority_ratios

        if self._label_ratios or self._minority_ratios:
            self._create_monitoring_metrics()


    def _create_monitoring_metrics(self):
        self._ratios = ["val"]

        self._ratios.extend([
            f"{label_ratio}_{spur_ratio}"
            for label_ratio in self._label_ratios
            for spur_ratio in self._minority_ratios
        ])

        for ratio in self._ratios:
            self.accuracy[ratio] = torchmetrics.Accuracy(task="multiclass", num_classes=self._num_classes)
            self.accuracy_wg[ratio] = WorstGroupAccuracy(num_groups=self._num_groups)
            self.loss[ratio] = torchmetrics.MeanMetric()
            self.loss_wg[ratio] = WorstGroupLoss(num_groups=self._num_groups)

            setattr(self, f"{ratio}_accuracy", self.accuracy[ratio])
            setattr(self, f"{ratio}_accuracy_wg", self.accuracy_wg[ratio])
            setattr(self, f"{ratio}_loss", self.loss[ratio])
            setattr(self, f"{ratio}_loss_wg", self.loss_wg[ratio])
    
    def sequence_forward(self, sequence):
        emb_sequence = self.embedding(sequence)
        transformer_output = self.backbone(inputs_embeds=emb_sequence).last_hidden_state
        pred_y = self.out(transformer_output)[:, ::2, :]

        return pred_y
    
    def forward(self, context, query, only_query=True):
        """
        context: (x, y) pair of vectors
        """
        sequence = construct_sequence(context, query)
        pred_y = self.sequence_forward(sequence).argmax(dim=-1)

        if only_query:
            return pred_y[:, -1]
        
        return pred_y.squeeze()

    def _create_labels_emb(self):
        if self._label_emb_mode == "opposite": # For binary
            if self._num_classes != 2:
                raise AttributeError("the 'opposite' mode can only be implemented for binary classification")

            label_vector = torch.rand(self._n_dims, dtype=torch.float32) * 2 # TODO: Automate (*2 manually add DINO's scale)
            labels = nn.Parameter(torch.stack([label_vector, -label_vector], dim=0), requires_grad=False) # NOTE: For binary classification
        
        if self._label_emb_mode == "random":
            labels = nn.Parameter(torch.rand((self._num_classes, self._n_dims), dtype=torch.float32) * 2, requires_grad=False) # NOTE: For binary classification

        return labels
    
    def _calculate_metrics(self, pred_y, y, groups, set_name):
        example_losses = self.criterion(pred_y, y)
        loss = torch.mean(example_losses)
        
        self.loss[set_name].update(loss.cpu())
        self.loss_wg[set_name].update(example_losses.cpu(), groups.cpu())
        self.accuracy[set_name].update(pred_y.cpu(), y.cpu())
        self.accuracy_wg[set_name].update(pred_y.cpu(), y.cpu(), groups.cpu())

        self.log(f"{set_name}_loss", self.loss[set_name], on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        self.log(f"{set_name}_loss_wg", self.loss_wg[set_name], on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        self.log(f"{set_name}_accuracy", self.accuracy[set_name], on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        self.log(f"{set_name}_accuracy_wg", self.accuracy_wg[set_name], on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)

        return loss

    def _eval_step(self, batch, set_name, *args, **kwargs):
        context, query, y, c = batch
        groups = y * self._num_confounders + c

        sequence = construct_sequence(context, query, self.labels)
        pred_y = self.sequence_forward(sequence)
        
        loss = self._calculate_metrics(pred_y[:, -1, :], y.squeeze(), groups.squeeze(), set_name)
        return loss
    
    def training_step(self, batch, *args, **kwargs):
        x, y, c = batch
        y = randomly_swap_labels(y) # randomly swap (0, 1)
        groups = y * self._num_confounders + c

        sequence = combine(x, y, self.labels)
        pred_y = self.sequence_forward(sequence)

        loss = self._calculate_metrics(pred_y.view(-1, 2), y.view(-1), groups.view(-1), "train") # flatten all batches

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx = None, *args, **kwargs):
        set_name = self._ratios[dataloader_idx]

        return self._eval_step(batch, set_name, *args, **kwargs)

    def test_step(self, batch, batch_idx, dataloader_idx = None, *args, **kwargs):
        return self._eval_step(batch, "test", *args, **kwargs)
    
    def configure_optimizers(self):
        target = self._optimizer_conf.pop('target')
        optimizer_conf = dict(**self._optimizer_conf, params=self.parameters())
        optimizer = instantiate(optimizer_conf, _target_=target)

        return optimizer
    
    def _on_epoch_end(self, set_name):
        for i in range(self._num_groups):
            gi_loss = self.loss_wg[set_name].compute_all_groups()[i]
            self.log(f"{set_name}_loss_g{i}", gi_loss, on_step=False, on_epoch=True)
            gi_acc = self.accuracy_wg[set_name].compute_all_groups()[i]
            self.log(f"{set_name}_accuracy_g{i}", gi_acc, on_step=False, on_epoch=True)

    def on_train_epoch_end(self):
        self._on_epoch_end("train")

    def on_validation_epoch_end(self):
        self._on_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_epoch_end("test")

    def on_before_optimizer_step(self, optimizer):
        norm = grad_norm(self, norm_type=2)['grad_2.0_norm_total']
        self.log('gradient norm', norm, on_step=True, on_epoch=False, prog_bar=False)