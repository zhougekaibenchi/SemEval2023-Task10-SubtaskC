#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/20 17:22
# @Author  : ZMY
import sys
sys.path.append("..")
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.dice_loss import DiceLoss
from utils.focal_loss import FocalLoss
from utils.function_utils import DynamicRoutingAggregator
from transformers import AutoModel, AutoConfig
from pytorch_metric_learning.losses import NTXentLoss, SupConLoss
from pytorch_metric_learning.distances import DotProductSimilarity
import warnings
warnings.filterwarnings('ignore')

class ClassifierModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(ClassifierModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.config = AutoConfig.from_pretrained(self.args.pretrain_model_path, gradient_checkpointing=True)
        self.model = AutoModel.from_pretrained(
            args.pretrain_model_path,
            from_tf=bool(".ckpt" in self.args.pretrain_model_path),
            config=self.config,
            cache_dir=self.args.cache_dir)
        self.model.resize_token_embeddings(len(tokenizer))

        self.class_nums = args.class_nums

        if self.args.mutisample_dropout == True:
            self.dropout_ops = nn.ModuleList(
                nn.Dropout(self.args.dropout_rate) for _ in range(self.args.dropout_num)
            )
        else:
            self.dropout_ops = nn.Dropout(self.args.dropout)
        # Max Avg Pooling
        self.layer_norm_MA = nn.LayerNorm(self.config.hidden_size*2)
        self.fc_MA = nn.Linear(self.config.hidden_size*2, self.class_nums)
        self.fc = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * 2)
        self._init_weights(self.layer_norm_MA)
        self._init_weights(self.fc_MA)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        # 初始化weight权重
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def loss_fn(self, logits, labels, embeddings):
        if self.args.loss_fct_name == "CrossEntropy loss":
            loss_fct = nn.CrossEntropyLoss(
                weight=None,
            )

        else:
            raise ValueError("unsupported loss function: {}".format(self.args.use_contrastive_loss))

        loss = loss_fct(logits, labels.long())
        if self.args.contrastive_loss is not None:
            if self.args.contrastive_loss == "NTXent loss":
                loss_fct_contrast = NTXentLoss(
                    temperature=self.args.contrastive_temperature,
                    distance=DotProductSimilarity(),
                )
            elif self.args.contrastive_loss == "SupCon loss":
                loss_fct_contrast = SupConLoss(
                    temperature=self.args.contrastive_temperature,
                    distance=DotProductSimilarity(),
                )
            else:
                raise ValueError("unsupported contrastive loss function: {}".format(self.args.use_contrastive_loss))

            if self.args.what_to_contrast == "sample":
                labels = labels.view(-1)
            elif self.args.what_to_contrast == "sample_and_class_embeddings":
                embeddings = torch.cat(
                    [embeddings, self.fc.weight],
                    dim=0
                )
                labels = torch.cat(
                    [
                        labels.view(-1),
                        torch.arange(0, self.args.num_labels_level_2).to(self.args.device)
                    ],
                    dim=-1
                )
            else:
                raise ValueError("unsupported contrastive features: {}".format(self.args.what_to_contrast))

            contra_loss = loss_fct_contrast(
                embeddings,
                labels
            )
            loss = loss + self.args.contrastive_loss_weight * contra_loss
        return loss

    def model_mutisample_dropout(self, x):
        logits = None
        for i, dropout_op in enumerate(self.dropout_ops):
            if i == 0:
                out = dropout_op(x)
                logits = self.fc_MA(out)
            else:
                temp_out = dropout_op(x)
                temp_logits = self.fc_MA(temp_out)
                logits += temp_logits

        if self.args.dropout_action:
            logits = logits / self.args.dropout_num

        return logits

    def replace_masked_values(self, tensor, mask, replace_with) -> torch.Tensor:
        """
        Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
        to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
        won't know which dimensions of the mask to unsqueeze.

        This just does ``tensor.masked_fill()``, except the pytorch method fills in things with a mask
        value of 1, where we want the opposite.  You can do this in your own code with
        ``tensor.masked_fill((1 - mask).byte(), replace_with)``.
        """
        if tensor.dim() != mask.dim():
            raise ValueError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
        return tensor.masked_fill((1 - mask).byte(), replace_with)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # MAXPOOLING
        if attention_mask is not None:
            input_tensors = self.replace_masked_values(
                outputs[0], attention_mask.unsqueeze(2), -1e7
            )

        input_max_pooled = torch.max(input_tensors, 1)[0]
        # AVGPOOLING
        if attention_mask is not None:
            input_tensors = self.replace_masked_values(
                input_tensors, attention_mask.unsqueeze(2), 0
            )
        tokens_avg_pooled = torch.mean(input_tensors, 1)
        mean_max_embeddings = torch.cat((input_max_pooled, tokens_avg_pooled), 1)
        # mean_max_embeddings = self.fc(mean_max_embeddings)
        mean_max_embeddings = self.layer_norm_MA(mean_max_embeddings)
        logits = self.model_mutisample_dropout(mean_max_embeddings)

        if labels is not None:
            loss = self.loss_fn(logits, labels, mean_max_embeddings)
        else:
            loss = 0
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output



