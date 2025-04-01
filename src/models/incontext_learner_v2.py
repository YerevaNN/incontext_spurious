"""In-context learning transformer implementation.

We use the following naming conventions:
  B: batch size
  L: sequence length
  C: number of context examples per class (class_context_size)
  D: model dimensionality
  Q: number of queries
"""
from typing import Optional, Tuple, Union

from hydra.utils import instantiate
from transformers.models.gptj import GPTJModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torchmetrics

from src.utils.custom_metrics import MinorityMajorityAccuracy, GroupAccuracy, WorstGroupAccuracy

torch.set_float32_matmul_precision('high')


class GPTJModelV2(GPTJModel):
    def __init__(self, config):
        super().__init__(config)
        for param in self.wte.parameters():
            param.requires_grad = False

    # @torch.compile(fullgraph=True)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        query_indices: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        assert past_key_values is None
        past_key_values = tuple([None] * len(self.h))

        # Constructing positions ids
        assert position_ids is None
        assert query_indices is not None
        seq_len = input_shape[-1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        query_indices = query_indices.to(dtype=torch.long)
        query_mask = torch.zeros(seq_len + 1, dtype=torch.long)
        query_mask[1 + query_indices] = 1
        position_ids -= torch.cumsum(query_mask, dim=0).to(device)[:seq_len]
        position_ids = position_ids.unsqueeze(0)

        # Constructing attention_mask tensor of shape [batch_size, num_heads, seq_len, seq_len],
        # which is going to be added after masking out non-causal pairs. Therefore, we just need
        # to prepare the part where we disallow any token to attend to any query token, unless it
        # is a query token attending to itself.
        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")
        attention_mask = torch.zeros((seq_len, seq_len),
                                     dtype=self.dtype,
                                     device=device)
        attention_mask[:, query_indices] = torch.finfo(self.dtype).min / 2
        attention_mask.fill_diagonal_(0)
        attention_mask.tril_()
        attention_mask = attention_mask.reshape((1, 1, seq_len, seq_len))#.tile((batch_size, 1, 1, 1))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_attention_heads x N x N
        # head_mask has shape n_layer x batch x num_attention_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states=hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

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
                v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


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
                 network: GPTJModelV2,
                 loss_fn,
                 val_sets,
                 dataset_name: str,
                 optimizer_conf=None,
                 scheduler_conf=None,
                 input_layer_norm: bool = False,
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
