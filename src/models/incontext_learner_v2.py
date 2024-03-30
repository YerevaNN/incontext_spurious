from typing import Optional, Tuple, Union

from transformers.models.gptj import GPTJModel
from transformers.modeling_outputs import BaseModelOutputWithPast
import torch
import torch.nn as nn

from src.models import InContextLearner


class GPTJModelV2(GPTJModel):
    def __init__(self, config):
        super().__init__(config)

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
        position_ids = position_ids.unsqueeze(0)
        query_indices = query_indices.to(dtype=torch.long)
        query_mask = torch.zeros(seq_len, dtype=torch.long)
        query_mask[query_indices] = 1
        position_ids -= torch.cumsum(query_mask, dim=0)

        # Constructing attention_mask tensor of shape [batch_size, num_heads, seq_len, seq_len],
        # which is going to be added after masking out non-causal pairs. Therefore, we just need
        # to prepare the part where we disallow any token to attend to any query token, unless it
        # is a query token attending to itself.
        assert not self._use_flash_attention_2
        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")
        attention_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=self.dtype)
        attention_mask[:, :, :, query_indices] = torch.finfo(self.dtype).min
        for q_idx in query_indices:
            attention_mask[:, :, q_idx, q_idx] = 0

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
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class InContextLearnerV2(InContextLearner):
    """In-context learner with different query prediction at each position.

    It can handle sequences of the following form: [x1, y1, q1, x2, y2, q2, ..., xn, yn, qn].
    I.e., spurious_setting is either 'sum' or 'no spurious' ('separate_token' is not allowed).
    """

    def __init__(self,
                 network: GPTJModelV2,
                 loss_fn,
                 val_sets,
                 spurious_setting: str,
                 dataset_name: str,
                 optimizer_conf = None,
                 scheduler_conf = None
                 ):
        """
        Args:
            network: The neural network to be used.
            loss_fn: The loss function for training.
            val_sets: A list of validation dataset names.
            spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
                                    Options include 'separate_token'(x,c) and 'sum'(x+c).
            dataset_name (str): Name of the dataset.
            optimizer_conf: Configuration dictionary for the optimizer.
            scheduler_conf: Configuration dictionary for the scheduler.
        """
        assert spurious_setting in ['sum', 'no_spurious']
        super(InContextLearnerV2, self).__init__(
            network=network,
            loss_fn=loss_fn,
            val_sets=val_sets,
            spurious_setting=spurious_setting,
            dataset_name=dataset_name,
            optimizer_conf=optimizer_conf,
            scheduler_conf=scheduler_conf,
        )

    def forward(self, input_embeds, *args, **kwargs):
        """
        Defines the forward pass for the model.

        Args:
            input_embeds: The input embeddings for the model.

        Returns:
            The output predictions of the model.
        """
        seq_len = input_embeds.shape[1]  # (batch_size, seq_len, hidden_dim)
        query_indices = torch.arange(2, seq_len, 3)

        out = self._network(
            inputs_embeds=input_embeds,
            query_indices=query_indices,
        ).last_hidden_state

        pred_embeddings = out[:, query_indices]
        pred_y = self._fc(pred_embeddings)
        return pred_y

    def _step(self, batch, set_name):
        """
        A step for training or validation.

        Args:
            batch: The batch of data for the step.
            set_name: The name of the dataset (e.g., 'train', 'val_inner', ...).

        Returns:
            The loss for the batch.
        """
        # TODO(hrayr): we need to implement corresponding datasets
        (input_seq, context_spurious_labels, context_class_labels,
         query_spurious_labels, query_class_labels,
         input_image_indices, query_image_indices) = batch

        pred_y_logit = self.forward(input_seq).squeeze()
        loss = self._loss_fn(pred_y_logit, query_class_labels.float())

        pred_y = nn.functional.sigmoid(pred_y_logit)
        last_pred_y = pred_y[:, -1]
        last_class_labels = query_class_labels[:, -1]
        last_spurious_class = query_spurious_labels[:, -1]

        self.accuracy[set_name].update(last_pred_y, last_class_labels)
        for min_maj_metric in [self.accuracy_minority[set_name],
                               self.accuracy_majority[set_name]]:
            min_maj_metric.update(
                query_prediction_batch=last_pred_y,
                query_target_batch=last_class_labels,
                query_spurious_batch=last_spurious_class,
                context_targets_batch=context_class_labels,
                context_spurious_vals_batch=context_spurious_labels,
            )

        self.log(f"{set_name}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{set_name}_accuracy", self.accuracy[set_name], on_step=False, on_epoch=True)
        self.log(f"{set_name}_accuracy_minority", self.accuracy_minority[set_name], on_step=False, on_epoch=True)
        self.log(f"{set_name}_accuracy_majority", self.accuracy_majority[set_name], on_step=False, on_epoch=True)

        if self._dataset_name == "waterbirds_emb_contexts":
            for i in range(4):
                self.group_accuracies[i][set_name].update(
                    query_prediction_batch=last_pred_y,
                    query_target_batch=last_class_labels,
                    query_spurious_batch=last_spurious_class)
                self.log(f"{set_name}_group_{i}_accuracy", self.group_accuracies[i][set_name], on_step=False,
                         on_epoch=True)

        return loss