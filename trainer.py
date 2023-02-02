#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : stce

import random
import os
import json
import math
import time
import torch
import random
import logging
import warnings
import scipy as sp
import numpy as np
import pandas as pd
from __init__ import *
from scipy import stats
from sklearn import metrics
from tqdm import tqdm, trange
from utils import function_utils
from utils.lookahead import Lookahead
from utils.at_training import FGM, PGD, AWP
from utils.function_utils import make_optimizer, make_scheduler
from sklearn.metrics import f1_score, classification_report
warnings.filterwarnings('ignore')


class Trainer(object):
    def __init__(self, args, train_loader, valid_loader, fold):
        self.args = args
        self.train_dataloader = train_loader
        self.valid_dataloader = valid_loader
        self.fold = fold

    def init_training(self, model):

        if not os.path.exists(self.args.model_save_path):
            os.makedirs(self.args.model_save_path)
        num_training_steps = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps) * self.args.num_epochs
        num_warmup_steps = int(self.args.warmup_ratio * num_training_steps)
        print(f"Total Training Steps: {num_training_steps}, Total Warmup Steps: {num_warmup_steps}")
        optimizer = Lookahead(make_optimizer(self.args, model), k=5, alpha=0.5)
        scheduler = make_scheduler(optimizer, self.args, decay_name=self.args.decay_name, t_max=num_training_steps, warmup_steps=num_warmup_steps)
        return (self.train_dataloader, self.valid_dataloader, optimizer, num_training_steps, scheduler, model)


    def train_model(self, model):
        # (1) 数据,日志,属性初始化
        train_dataloader, valid_dataloader, optimizer, num_training_steps, scheduler, model = self.init_training(model)
        start_time = time.time()

        logger.info("***** Running training *****")
        logger.info("  Num Train examples = %d", len(train_dataloader))
        logger.info("  Num Valid examples = %d", len(valid_dataloader))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Train batch size = %d", self.args.train_batch_size)
        logger.info("  Batch interval = %d", self.args.batch_interval)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        model_output_dir = os.path.join(self.args.model_save_path, f"pytorch_model_{self.fold}.bin")
        early_taskA = function_utils.EarlyStopping(self.args, path=model_output_dir)
        train_epoch = trange(int(self.args.num_epochs), desc="Epoch")
        losses = function_utils.AverageMeter()
        adv_trainer = AWP(model, optimizer, self.args.apex, adv_lr=self.args.awp_lr, adv_eps=self.args.awp_eps)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.apex)
        flag = True


        # (2)训练主循环开始
        for epoch in train_epoch:
            # (2-1) 每个epoch初始化设置
            if flag == False:
                break
            # (2-2) 进入迭代
            train_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, data in enumerate(train_iterator):
                # (2-2-1) 正向传播
                model.train()
                input_ids, attention_mask, token_type_ids, label = data['input_ids'].to(self.args.device), \
                                                                       data['attention_mask'].to(self.args.device), \
                                                                       data['token_type_ids'].to(self.args.device), \
                                                                       data["label"].to(self.args.device),
                with torch.cuda.amp.autocast(enabled=self.args.apex):
                    output = model(input_ids, attention_mask, token_type_ids, label)
                    loss, logits = output[0], output[1]

                # (2-2-2) 统计部分训练指标
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                # (2-2-3) 正常反向传播
                scaler.scale(loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                #(2-2-4) 对抗训练
                if self.args.awp:
                    if epoch >= self.args.nth_awp_start_epoch and epoch <= self.args.nth_awp_end_epoch:
                        with torch.cuda.amp.autocast(enabled=self.args.apex):
                            adv_trainer._save()
                            adv_trainer._attack_step()
                            output = model(input_ids, attention_mask, token_type_ids, label)
                            loss, _ = output[0], output[1]
                            model.zero_grad()
                        scaler.scale(loss).backward()
                        adv_trainer._restore()

                # (2-2-5) 更新loss, scaler, optimizer, scheduler参数
                losses.update(loss.item(), input_ids.size(0))
                # if (step + 1) % self.args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad()
                scheduler.step()

                # (2-2-6) 是否进入Evaluation阶段
                if step % self.args.batch_interval == 0 and step > 0 or step + 1 == len(train_dataloader):
                    # (1) 计算打印训练集合指标
                    time_dif = function_utils.get_time_dif(start_time)
                    _s = str(len(str(len(train_dataloader.sampler))))
                    _lr = sum(scheduler.get_lr()) / len(scheduler.get_lr())

                    logger.info("*" * 50)
                    logger.info("TRAIN RECORD:")
                    logger.info(('Fold: {}').format(self.fold))
                    logger.info(('Epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch + 1, step + 1,
                                                                                           len(train_dataloader),
                                                                                           100 * (step + 1) / len(
                                                                                               train_dataloader)))
                    logger.info(('Train Loss: {loss: >.5f}({avg_loss:.5f})').format(loss=loss.item(), avg_loss=losses.avg,))
                    logger.info(('Grad: {grad_norm: >.3f}').format(grad_norm=grad_norm))
                    logger.info(('LR: {lr: >.10f}').format(lr=_lr))
                    logger.info(('Time: {}').format(time_dif))
                    logger.info("*" * 50)

                    # (3) 测试集计算
                    valid_loss, score = self.evaluate(model, valid_dataloader, epoch)

                    # (4) EarlyStop，保存模型
                    early_taskA(1 - score, model)
                    #early_taskA(valid_loss, model)
                    if early_taskA.early_stop:
                        flag = False
                        logger.info(f"========== fold: {self.fold} result ==========")
                        logger.info({f"[fold{self.fold}] valid best score taskC": 1+early_taskA.best_score})
                        break
            if flag == False:
                logger.info({f"[fold{self.fold}] valid best score taskC": 1 + early_taskA.best_score})
                break

    def record(self, logits, label, predit_sum, label_sum):
        logits = torch.softmax(logits, dim=1)
        label = label.data.cpu().detach().numpy()
        out_sort = torch.argmax(logits, dim=1).cpu().detach().numpy()

        label_sum = np.hstack((label_sum, label))
        predit_sum = np.hstack((predit_sum, out_sort))
        return predit_sum, label_sum

    def metric_evaluation(self, predict, label):
        score = metrics.f1_score(label, predict, average='macro')
        return score

    def evaluate(self, model, valid_dataloader, epoch):

        model.eval()
        start_time = time.time()
        with torch.no_grad():
            losses = function_utils.AverageMeter()
            predit_sum, label_sum, id_sum = [], [], []
            for step, data in enumerate(valid_dataloader):
                input_ids, attention_mask, token_type_ids, label = data['input_ids'].cuda(), \
                                                                   data['attention_mask'].cuda(), \
                                                                   data['token_type_ids'].cuda(), \
                                                                   data['label'].cuda(), \

                output = model(input_ids, attention_mask, token_type_ids, label)
                loss, logits = output[0], output[1]
                losses.update(loss.item(), input_ids.size(0))

                out_label = label.data.cpu().detach().numpy()
                logits = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().detach().numpy()

                label_sum = np.hstack((label_sum, out_label))
                predit_sum = np.hstack((predit_sum, logits))

            time_dif = function_utils.get_time_dif(start_time)
            _s = str(len(str(len(valid_dataloader))))
            score = metrics.f1_score(label_sum, predit_sum, average='macro')
            logger.info("*" * 50)
            logger.info("Valid RECORD:")
            logger.info(('Fold: {}').format(self.fold))
            logger.info(('Epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch + 1, step + 1,
                                                                                       len(valid_dataloader),
                                                                                       100 * (step + 1) / len(
                                                                                       valid_dataloader)))
            logger.info(('Valid Loss: {loss: >.5f}({avg_loss:.5f})').format(loss=loss.item(), avg_loss=losses.avg))
            logger.info(('Macro F1: {macro_f1: >.4f}').format(macro_f1=score))
            logger.info(('Time: {}').format(time_dif))
            logger.info("*" * 50)
        return loss, score
