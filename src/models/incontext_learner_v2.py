"""In-context learning transformer implementation.

We use the following naming conventions:
  B: batch size
  L: sequence length
  C: number of context examples per class (class_context_size)
  D: model dimensionality
  Q: number of queries
"""
from typing import Optional, Tuple, Union, Any

from hydra.utils import instantiate
from transformers.models.gptj import GPTJModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from pytorch_lightning import LightningModule
import logging
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import transformers

from src.utils.custom_metrics import MinorityMajorityAccuracy, GroupAccuracy, WorstGroupAccuracy

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

log = logging.getLogger(__name__)


class GPTJModelV2(GPTJModel):
    def __init__(self, config):
        assert transformers.__version__ == '4.48.0'
        super().__init__(config)
        for param in self.wte.parameters():
            param.requires_grad = False

    @torch.compile(fullgraph=True)
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        query_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        current_stackformer_depth: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = inputs_embeds.shape[0]
        seq_length = inputs_embeds.shape[1]

        # Construct positions ids
        assert query_indices is not None
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=inputs_embeds.device)
        query_indices = query_indices.to(dtype=torch.long)
        query_mask = torch.zeros(seq_length + 1, dtype=torch.long)
        query_mask[1 + query_indices] = 1
        position_ids -= torch.cumsum(query_mask, dim=0).to(inputs_embeds.device)[:seq_length]
        position_ids = position_ids.unsqueeze(0).repeat((batch_size, 1))  # [batch_size, seq_length]

        # Construct causal mask
        causal_mask = self._prepare_4d_attention_mask(
            sequence_length=seq_length,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
            batch_size=batch_size,
            query_indices=query_indices,
        )

        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = (-1, seq_length, hidden_states.size(-1))

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            # Stop if current depth of StackFormer is reached
            if (current_stackformer_depth is not None) and i > current_stackformer_depth:
                break
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)

                # Ensure that attention_mask is always on the same device as hidden_states
                if causal_mask is not None:
                    causal_mask = causal_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, None, all_hidden_states, all_self_attentions] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    @staticmethod
    def _prepare_4d_attention_mask(
        sequence_length: int,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
        query_indices: Optional[torch.FloatTensor] = None,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)`.
        It is a standard causal attention with a specific rule that no token attends to an
        intermediate query, unless it is that intermediate query itself.

        Args:
            sequence_length (`int`):
                The sequence length being processed.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, sequence_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        else:
            assert sequence_length == 1
            causal_mask[0, 0] = 0

        if query_indices is not None:
            correction_for_intermediate_queries = torch.zeros(
                (sequence_length, sequence_length),
                dtype=dtype,
                device=device
            )
            correction_for_intermediate_queries[:, query_indices] = min_dtype
            correction_for_intermediate_queries.fill_diagonal_(0)
            correction_for_intermediate_queries.tril_()
            causal_mask += correction_for_intermediate_queries

        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        return causal_mask


class InContextLearnerV2(LightningModule):
    """In-context learner with different query prediction at each position.

    This transformer expects a list of tokens some of which are query tokens, and produces predictions on queries.
    Importantly, query tokens are not attended to by tokens on their right. For example, this ICL transformer can
    handle a sequences like this:
    (a) [x1, y1, q1, x2, y2, q2, ..., xn, yn, qn]
    (b) [x1, c1, y1, q1, x2, c2, y2, q2, ..., xn, cn, yn, qn]
    where xi (context example) and qi (query example) are expected to be representations, likely containing information
    about spurious features; ci are binary spurious features encoded in R^D; and yi are binary labels encoded in R^D.
    """

    def __init__(self,
                 embedding_size: int,
                 network_config: transformers.GPTJConfig,
                 loss_fn,
                 val_sets,
                 dataset_name: str,
                 optimizer_conf=None,
                 scheduler_conf=None,
                 input_layer_norm: bool = False,
                 stackformer: bool = False,
                 stackformer_block_size: int = 2,
                 ):
        """
        Args:
            embedding_size: The size of image representation.
            network: The neural network to be used.
            loss_fn: The loss function for training.
            val_sets: A list of validation dataset names.
            dataset_name (str): Name of the dataset.
            optimizer_conf: Configuration dictionary for the optimizer.
            scheduler_conf: Configuration dictionary for the scheduler.
        """
        super(InContextLearnerV2, self).__init__()

        if input_layer_norm:
            self._input_ln = nn.LayerNorm(embedding_size, eps=1e-5)
        else:
            self._input_ln = None

        self._network_config = network_config
        network = GPTJModelV2(network_config)

        if embedding_size != network.embed_dim:
          self._proj = nn.Linear(embedding_size, network.embed_dim)
        else:
          self._proj = None

        self._network = network
        self._fc = nn.Linear(network.embed_dim, 1)

        self._loss_fn = loss_fn
        self._val_sets = [f"val_{x}" for x in val_sets] if val_sets else ['val']
        self._dataset_name = dataset_name
        self._optimizer_conf = optimizer_conf
        self._scheduler_conf = scheduler_conf

        # StackFormer related configs
        self._stackformer = stackformer
        self._stackformer_block_size = stackformer_block_size
        self._stackformer_num_stages, r = divmod(network_config.n_layer, stackformer_block_size)
        assert r == 0, "Number of layers should be divisible by StackFormer block size."
        if stackformer:
            self._current_stackformer_depth = stackformer_block_size
        else:
            self._current_stackformer_depth = None

        # Metrics
        self.accuracy = dict()
        self.accuracy_minority = dict()
        self.accuracy_majority = dict()

        if dataset_name in ["waterbirds_emb_contexts", "celeba_emb_contexts", "multinli_emb_contexts",
                            "camelyon17_emb_contexts", "spawrious_emb_contexts", "civilcomments_emb_contexts"]:
            self.group_accuracies = [dict() for _ in range(4)]
            self.worst_group_accuracy = dict()

        self._initialize_metrics()

    def forward(
            self,
            input_embeds: torch.Tensor,
            query_indices: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_embeds: Torch tensor of shape (B, L, D).
            query_indices: Torch tensor of shape (Q,) describing query token positions.

        Returns: a torch tensor of shape (B, Q, 1) consisting of query prediction logits.
        """
        if self._input_ln is not None:
            input_embeds = self._input_ln(input_embeds)
        if self._proj is not None:
            input_embeds = self._proj(input_embeds)
        out = self._network(
            inputs_embeds=input_embeds,
            query_indices=query_indices,
            current_stackformer_depth=self._current_stackformer_depth,
            # output_attentions=True,
        )
        out = out.last_hidden_state
        pred_embeddings = out[:, query_indices]
        pred_y = self._fc(pred_embeddings)
        return pred_y

    def _step(self, batch, set_name):
        """A step for training or validation.

        Args:
            batch: The batch of data for the step. Should be (input_seq, context, queries, query_indices).
                input_seq should be a tensor of shape (B, L, D). context and queries should be tensors of shape
                (B, 2*C, 3) describing context/query examples with (id, spurious_label, class_label) triplets.
                query_indices should be a tensor of shape (B, Q) with equal rows.
            set_name: The name of the dataset (e.g., 'train', 'val_inner', ...).

        Returns:
            The loss for the batch.
        """
        input_seq, context, queries, query_indices = batch

        pred_y_logit = self.forward(input_seq, query_indices[0]).squeeze()
        query_class_labels = queries[:, :, 2]
        loss = self._loss_fn(pred_y_logit, query_class_labels.float())
        # loss = self._loss_fn(pred_y_logit[:, 128:], query_class_labels[:, 128:].float())

        with torch.no_grad():
            last_pred_y = nn.functional.sigmoid(pred_y_logit[:, -1])
            last_spurious_class = queries[:, -1, 1]
            last_class_labels = queries[:, -1, 2]

            self.accuracy[set_name].update(last_pred_y, last_class_labels)

            for min_maj_metric in [self.accuracy_minority[set_name],
                                   self.accuracy_majority[set_name]]:
                min_maj_metric.update(
                    query_prediction_batch=last_pred_y,
                    query_target_batch=last_class_labels,
                    query_spurious_batch=last_spurious_class,
                    context_targets_batch=context[:, :, 2],
                    context_spurious_vals_batch=context[:, :, 1],
                )

            self.log(f"{set_name}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"{set_name}_accuracy", self.accuracy[set_name], on_step=False, on_epoch=True)
            self.log(f"{set_name}_accuracy_minority", self.accuracy_minority[set_name], on_step=False, on_epoch=True)
            self.log(f"{set_name}_accuracy_majority", self.accuracy_majority[set_name], on_step=False, on_epoch=True)

            if self._dataset_name in ["waterbirds_emb_contexts", "celeba_emb_contexts", "multinli_emb_contexts",
                                      "camelyon17_emb_contexts", "spawrious_emb_contexts", "civilcomments_emb_contexts"]:
                self.worst_group_accuracy[set_name].update(
                    preds=last_pred_y,
                    targets=last_class_labels,
                    spurious_labels=last_spurious_class,
                )
                self.log(f"{set_name}_worst_group_accuracy", self.worst_group_accuracy[set_name], on_step=False,
                         on_epoch=True)

                for i in range(4):
                    self.group_accuracies[i][set_name].update(
                        query_prediction_batch=last_pred_y,
                        query_target_batch=last_class_labels,
                        query_spurious_batch=last_spurious_class)
                    self.log(f"{set_name}_group_{i}_accuracy", self.group_accuracies[i][set_name], on_step=False,
                             on_epoch=True)

        return loss

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        set_name = self._val_sets[dataloader_idx]
        return self._step(batch, set_name)

    def configure_optimizers(self):
        """Configures the optimizers and learning rate schedulers.

        Returns:
            The optimizer and (optionally) the learning rate scheduler.
        """
        target = self._optimizer_conf.pop('target')
        optimizer_conf = dict(**self._optimizer_conf, params=self.parameters())
        optimizer = instantiate(optimizer_conf, _target_=target)

        if self._scheduler_conf.get('target', None) is None:
            return optimizer
        else:
            monitor = self._scheduler_conf.pop('monitor', None)
            interval = self._scheduler_conf.pop('interval', None)
            scheduler_target = self._scheduler_conf.pop('target')
            scheduler = instantiate(self._scheduler_conf, optimizer=optimizer, _target_=scheduler_target)

            ret_opt = dict(optimizer=optimizer,
                           lr_scheduler={"scheduler": scheduler, "monitor": monitor, "interval": interval})

            return ret_opt

    def _initialize_metrics(self):
        """Initializes metrics for training and validation."""

        for set_name in ["train"] + self._val_sets:
            if self._dataset_name in ["waterbirds_emb_contexts", "celeba_emb_contexts", "multinli_emb_contexts",
                                      "camelyon17_emb_contexts", "spawrious_emb_contexts", "civilcomments_emb_contexts"]:
                self.worst_group_accuracy[set_name] = WorstGroupAccuracy()
                for i in range(4):
                    self.group_accuracies[i][set_name] = GroupAccuracy(group=i)

            self.accuracy[set_name] = torchmetrics.Accuracy(task="binary")
            self.accuracy_minority[set_name] = MinorityMajorityAccuracy(group_type="minority")
            self.accuracy_majority[set_name] = MinorityMajorityAccuracy(group_type="majority")

            self._set_metric_attributes(set_name)

    def _set_metric_attributes(self, set_name):
        """Sets metric attributes for a given set name."""
        setattr(self, f"{set_name}_accuracy", self.accuracy[set_name])
        setattr(self, f"{set_name}_accuracy_minority", self.accuracy_minority[set_name])
        setattr(self, f"{set_name}_accuracy_majority", self.accuracy_majority[set_name])

        if self._dataset_name in ["waterbirds_emb_contexts", "celeba_emb_contexts", "multinli_emb_contexts",
                                  "camelyon17_emb_contexts", "spawrious_emb_contexts", "civilcomments_emb_contexts"]:
            setattr(self, f"{set_name}_worst_group_accuracy", self.worst_group_accuracy[set_name])
            for i in range(4):
                setattr(self, f"{set_name}_group_{i}_accuracy", self.group_accuracies[i][set_name])

    def on_after_backward(self):
        """Computes and logs gradient L2 norm."""
        grad_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        # Log it (use on_step=True if you want it per step)
        self.log('grad_norm', grad_norm, on_step=True, on_epoch=False, prog_bar=True)

    @torch.no_grad
    def _copy_layer_params_and_opt_states(self, source_layer_idx: int, target_layer_idx: int):
        param_dict = dict(self._network.named_parameters())
        opt_state = self.trainer.optimizers[0].state
        for k, v in param_dict.items():
            if k.startswith(f'h.{source_layer_idx}.'):
                k_target = k.replace(f'h.{source_layer_idx}.', f'h.{target_layer_idx}.')
                log.info(f'\tCopying parameters from {k} to {k_target}')
                param_dict[k_target].copy_(v)
                if v in opt_state:
                    log.info(f'\tCopying optimizer states from {k} to {k_target}')
                    opt_state[param_dict[k_target]] = opt_state[v]

    def _stackformer_stack(self):
        """Does one stacking step of StackFormer."""
        log.info("*"*80)
        log.info("Doing a StackFormer stacking operation")
        log.info(f"\tCurrent depth: {self._current_stackformer_depth}")
        log.info(f"\tBlock size: {self._stackformer_block_size}")

        current_n_layers = self._current_stackformer_depth
        target_n_layers = current_n_layers + self._stackformer_block_size
        current_n_blocks, r = divmod(current_n_layers, self._stackformer_block_size)
        assert r == 0
        middle_block_idx = current_n_blocks // 2

        for layer_idx in range(
            target_n_layers - 1,
            (middle_block_idx + 1) * self._stackformer_block_size - 1,
            -1):

            self._copy_layer_params_and_opt_states(
                source_layer_idx=layer_idx - self._stackformer_block_size,
                target_layer_idx=layer_idx)

        self._current_stackformer_depth += self._stackformer_block_size
        log.info("*"*80)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        if not self._stackformer:
            return
        num_batches = self.trainer.num_training_batches  # TODO(hrayrh): do I need a special logic for DDP
        stackformer_stage_length = np.ceil(num_batches / self._stackformer_num_stages)
        if batch_idx > 0 and batch_idx % stackformer_stage_length == 0:
            self._stackformer_stack()
            torch._dynamo.reset()  # reset dynamo cache so that recompilations don't exceed cache size
