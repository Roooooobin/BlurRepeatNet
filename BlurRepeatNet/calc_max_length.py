import pandas as pd

data_cikm_train = pd.read_table(r"../datasets/cikm16/cikm16_train.txt")
data_cikm_test = pd.read_table(r"../datasets/cikm16/cikm16_test.txt")
cikm_train_session_max_len = data_cikm_train.groupby("SessionId")["ItemId"].count().max()
cikm_test_session_max_len = data_cikm_test.groupby("SessionId")["ItemId"].count().max()
print(cikm_train_session_max_len)
print(cikm_test_session_max_len)

data_rsc_train = pd.read_table(r"../datasets/rsc15/rsc15_train_64.txt")
data_rsc_test = pd.read_table(r"../datasets/rsc15/rsc15_test_64.txt")
rsc_train_session_max_len = data_rsc_train.groupby("SessionId")["ItemId"].count().max()
rsc_test_session_max_len = data_rsc_test.groupby("SessionId")["ItemId"].count().max()

print(rsc_train_session_max_len)
print(rsc_test_session_max_len)
#
# """
# 70
# 41
# 146
# 67
# """
import numpy as np
import pandas as pd
#
data_cikm_train = pd.read_table(r"../datasets/cikm16/cikm16_train_full.txt")
train_items = data_cikm_train["ItemId"].unique()
data_cikm_test = pd.read_table(r"../datasets/cikm16/cikm16_test.txt")
test_items = data_cikm_test["ItemId"].unique()
print(train_items.size)
mask = np.in1d(test_items, train_items)
print(mask.all())
mask = np.array([True, True, True])
print(mask.all())
data_rsc_train = pd.read_table(r"../datasets/rsc15/rsc15_train_64.txt")
print(data_rsc_train["ItemId"].unique().size)
# #
# # mapping = {123: 234, 234: 345}
# # np.save("mapping_cikm16_item_ids", mapping)
# # mapping = np.load("mapping_cikm16_item_ids.npy")
# # print(mapping)
