#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/1/31 11:46
# @Author  : stce
import pandas as pd
import numpy as np

data_1 = pd.read_csv("1/test_logits.csv")
data_2 = pd.read_csv("2/test_logits.csv")
data_3 = pd.read_csv("3/test_logits.csv")
data_4 = pd.read_csv("4/test_logits.csv")
data_5 = pd.read_csv("5/test_logits.csv")
data_6 = pd.read_csv("6/test_logits.csv")
data_7 = pd.read_csv("7/test_logits.csv")
data_8 = pd.read_csv("8/test_logits.csv")
data_9 = pd.read_csv("9/test_logits.csv")
data_10 = pd.read_csv("10/test_logits.csv")

data = (data_1 + data_2 + data_3 + data_4 + data_5 + data_8 + data_9 + data_10)/8


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
pd.DataFrame(submission).to_csv("stce_2.csv", sep=',', index=False)

# data_21 = pd.read_csv("11/test_logits.csv")
# data_22 = pd.read_csv("22/test_logits.csv")
# data_23 = pd.read_csv("12/test_logits.csv")
# data_24 = pd.read_csv("13/test_logits.csv")
# data_25 = pd.read_csv("25/test_logits.csv")
#
# data = (data_21+data_23+data_24)/3
#0.5206


# data_26 = pd.read_csv("26/test_logits.csv")
# data_27 = pd.read_csv("27/test_logits.csv")
# data_28 = pd.read_csv("28/test_logits.csv")
# data = (data_26+data_27+data_28)/3