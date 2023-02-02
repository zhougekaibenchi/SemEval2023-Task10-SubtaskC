#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/21 14:00
# @Author  : ZMY
import sys
sys.path.append("..")
import torch
import numpy as np
import torch.nn as nn
from utils.dice_loss import DiceLoss
from utils.focal_loss import FocalLoss
from transformers import AutoModel, AutoConfig
from pytorch_metric_learning.losses import NTXentLoss, SupConLoss
from pytorch_metric_learning.distances import DotProductSimilarity


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

        self.model.training = True
        self.model.resize_token_embeddings(len(tokenizer))

        self.class_nums = args.class_nums

        self.fc = nn.Linear(self.config.hidden_size, self.class_nums)
        self.layer_norm = nn.LayerNorm(self.config.hidden_size)

        if self.args.mutisample_dropout == True:
            self.dropout_ops = nn.ModuleList(
                nn.Dropout(self.args.dropout_rate) for _ in range(self.args.dropout_num)
            )
        else:
            self.dropout_ops = nn.Dropout(self.args.dropout)
        self._init_weights(self.layer_norm)
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

    def loss_fn(self, logits, labels, pooler_output):
        # todo weight
        if self.args.use_class_weights:
            weight = self.label_frequency
            weight = torch.from_numpy(np.array(weight)).cuda()
        else:
            weight = None

        if self.args.loss_fct_name == "CrossEntropy loss":
            loss_fct = nn.CrossEntropyLoss(
                weight=None,
            )
        elif self.args.loss_fct_name == "Focal loss":
            loss_fct = FocalLoss(
                gamma=self.args.focal_loss_gamma,
                alpha=weight,
                reduction="mean"
            )
        elif self.args.loss_fct_name == "Dice loss":
            loss_fct = DiceLoss(
                with_logits=True,
                smooth=1.0,
                ohem_ratio=0.8,
                alpha=0.01,
                square_denominator=True,
                index_label_position=True,
                reduction="mean"
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
                embeddings = pooler_output
                labels = labels.view(-1)
            elif self.args.what_to_contrast == "sample_and_class_embeddings":
                embeddings = torch.cat(
                    [pooler_output, self.fc.weight],
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
                logits = self.fc(out)
            else:
                temp_out = dropout_op(x)
                temp_logits = self.fc(temp_out)
                logits += temp_logits

        if self.args.dropout_action:
            logits = logits / self.args.dropout_num

        return logits

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        pooler_output = outputs["pooler_output"]
        pooler_output = self.layer_norm(pooler_output)
        logits = self.model_mutisample_dropout(pooler_output)

        if labels is not None:
            loss = self.loss_fn(logits, labels, pooler_output)
        else:
            loss = 0

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output