#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/22 14:00
# @Author  : ZMY

#################### SEMEVAL 2023 task 10  #################

import gc
import warnings
import argparse
from __init__ import *
from trainer import Trainer
from utils import task_utils
from utils import function_utils
from importlib import import_module
from sklearn.model_selection import StratifiedKFold

gc.enable()
warnings.filterwarnings('ignore')


def main(args):

    x = import_module('model.' + args.model_head)
    function_utils.seed_everything(args.seed)
    train_df = task_utils.read_dataset(args)
    test_df = task_utils.read_test_dataset(args)

    # 不同Task的划分方式应该是不同的 MultilabelStratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df[["text", "taskA_label", "taskB_label", "taskC_label"]],
                                                          train_df["taskC_label"]), start=0):

        train_loader, valid_loader, tokenizer = task_utils.get_train_val_dataloader(args, train_df, train_idx, val_idx)
        model = x.ClassifierModel(args, tokenizer).to(args.device)
        trainer = Trainer(args, train_loader, valid_loader, fold)
        trainer.train_model(model)


    test_dataloader, tokenizer = task_utils.get_test_dataloader(args, test_df)
    model = x.ClassifierModel(args, tokenizer).to(args.device)
    task_utils.test_model(args, model, test_dataloader, skf.n_splits)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SemEval2023 Task10 C Code')
    parser.add_argument('--model_head', type=str, default='attention',
                        help='MAIN HEAD_POOLING FUNCTION:'
                             'roberta:'
                                "(1) CLS_Pooling_F"
                                "(2) MAX_AVG_POOLING_F"
                                "(3) attention"
                             'deberta:'
                                "(1) MAX_AVG_POOLING_F"
                                "(2) attention")

    parser.add_argument('--class_nums', type=int, default=11)
    # ----------------------------------------------------------------------------------------------------------------------
    # Data and Result Path
    parser.add_argument("--train_df", default="data/official data/starting_ki/train_all_tasks.csv", type=str)
    parser.add_argument("--test_df", default="data/official data/dev_task_c_entries.csv", type=str, help="data/official data/dev_task_c_entries.csv")
    parser.add_argument("--model_save_path", default="output/", type=str, help="Model Saving Location")
    parser.add_argument("--submission_path", default="output/SUBMISSION_dev_task_c.csv", type=str, help="提交文件")

    # ----------------------------------------------------------------------------------------------------------------------
    # Optimizer and scheduler
    parser.add_argument('--gradient_accumulation_steps', type=float, default=1.0, help="Gradient Accumulation")
    parser.add_argument("--optimizer_type", default='AdamW', type=str,
                        help="Main Optimizer Type: (1) Adam(2) AdamW(3) LAMB(4) MADGRAD")
    parser.add_argument("--higher_optimizer", default="lookahead", type=str, help="((1) lookahead, (2)swa, (3) None")
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--epsilon", default=1e-8, type=float, help="")
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--decay_name", default='cosine_warmup', type=str, help="")
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--apex", default=True, type=bool, help="打开自动混合精度")

    # ----------------------------------------------------------------------------------------------------------------------
    # Model Hyperparameter For TASK4_1
    parser.add_argument("--batch_interval", default=60, type=float, help="Batch Interval for doing Evaluation")
    parser.add_argument('--logging_steps', type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--patience", default=6, type=int, help="Early-Stopping Batch Interval")
    parser.add_argument('--seed', type=int, default=405, help="random seed for initialization") #2020
    parser.add_argument("--dynamic_padding", default=False, type=int, help="(1) True;(2) False")
    parser.add_argument('--max_sequence_length', type=int, default=100,
                        help="Choose Max Length, If Dynamic Dadding is False")
    parser.add_argument("--train_batch_size", default=30, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=100, type=int, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=1.1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_epochs", default=20.0, type=float,
                        help="Total number of training epochs to perform.")

    # -----------------------------------------------------------------------------------------------------------------------
    # for multi-sample dropout
    parser.add_argument("--mutisample_dropout", default=True, type=str, help="multi-sample dropout rate")
    parser.add_argument("--dropout_rate", default=0.2, type=float, help="multi-sample dropout rate")
    parser.add_argument("--dropout_num", default=4, type=int, help="how many dropout samples to draw")
    parser.add_argument("--dropout_action", default="sum", type=str, help="sum, avg")

    # -----------------------------------------------------------------------------------------------------------------------
    # Deal with Unbalanced Sample Distribution
    parser.add_argument("--use_weighted_sampler", default=True, type=bool,
                        help="Different Loss Caculation Weight For Different lass")
    parser.add_argument("--use_class_weights", default=False, type=bool,
                        help="Different Loss Caculation Weight For Different Class")

    # ----------------------------------------------------------------------------------------------------------------------
    # Loss Function
    parser.add_argument("--loss_fct_name", type=str, default="CrossEntropy loss",
                        help="(1) CrossEntropy loss; (2) Focal loss; (3) Dice loss; (4) MSE loss")
    parser.add_argument("--focal_loss_gamma", default=2.0, type=float, help="gamma in focal loss")
    parser.add_argument("--contrastive_loss", default="NTXent loss", type=str, help="(1) NTXent loss (2) SupCon loss")
    parser.add_argument("--what_to_contrast", default="sample", type=str,
                        help="(1) sample; (2) sample_with_class_embeddings")
    parser.add_argument("--contrastive_loss_weight", default=0.2, type=float, help="loss weight for ntxent")
    parser.add_argument("--contrastive_temperature", default=0.5, type=float, help="temperature for contrastive loss")

    # ----------------------------------------------------------------------------------------------------------------------
    # Adversarial Training
    parser.add_argument("--awp", default=True, type=bool,)
    parser.add_argument("--awp_eps", default=1e-2, type=float)
    parser.add_argument("--awp_lr", default=1e-4, type=float)
    parser.add_argument("--nth_awp_start_epoch", default=1, type=int)
    parser.add_argument("--nth_awp_end_epoch", default=15, type=int)  # 20

    # ----------------------------------------------------------------------------------------------------------------------
    # Pretrain Model Select
    parser.add_argument('--model_name', type=str, default='deberta-v3-large-pretrain-2',
                        help='MAIN MODEL_TYPE:'
                             "(1) roberta-large"
                             "(2) microsoft/deberta-v3-large"

                             "预训练的模型"
                             "(1) roberta-large-pretrain-1"
                             "(2) deberta-v3-large-pretrain-1"
                             "(3) deberta-v3-large-pretrain-2"
                             "checkpoint-102312"

                        )

    parser.add_argument("--pretrain_model_path", default='deberta-v3-large-pretrain-1', type=str, help="预训练模型地址")
    parser.add_argument("--cache_dir", default='deberta-v3-large-pretrain-1', type=str, help="预训练模型地址")
    parser.add_argument("--hidden_size", default=1024, type=int, help="隐藏层")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    args = parser.parse_args()
    logger.info("************************** Args Statement *********************************")
    logger.info("Args: {}".format(args))
    main(args)