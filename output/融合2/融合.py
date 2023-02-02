#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/1/31 11:46
# @Author  : stce
import pandas as pd
import numpy as np


data_2 = pd.read_csv("12/test_logits.csv")
data_3 = pd.read_csv("13/test_logits.csv")
data_4 = pd.read_csv("14/test_logits.csv")

data_7 = pd.read_csv("17/test_logits.csv")
data_8 = pd.read_csv("18/test_logits.csv")

data_10 = pd.read_csv("20/test_logits.csv")


data = (data_2 + data_3 + data_4 + data_7 + data_8 + data_10)/6

test_df = np.argmax(np.array(data), axis=1).tolist()

label_dict_3 = {
    0: "1.1 threats of harm",
    1: "1.2 incitement and encouragement of harm",
    2: "2.1 descriptive attacks",
    3: "2.2 aggressive and emotive attacks",
    4: "2.3 dehumanising attacks & overt sexual objectification",
    5: "3.1 casual use of gendered slurs, profanities, and insults",
    6: "3.2 immutable gender differences and gender stereotypes",
    7: "3.3 backhanded gendered compliments",
    8: "3.4 condescending explanations or unwelcome advice",
    9: "4.1 supporting mistreatment of individual women",
    10: "4.2 supporting systemic discrimination against women as a group",
}

test_submission = []
for item in test_df:
    test_submission.append(label_dict_3[item])
submission = pd.read_csv("test_task_c_entries.csv").iloc[:, :1]
submission["label_pred"] = test_submission
pd.DataFrame(submission).to_csv("JMZ_2.csv", sep=',', index=False)