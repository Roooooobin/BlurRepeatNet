# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime as dt

PATH_TO_ORIGINAL_DATA = ""
PATH_TO_PROCESSED_DATA = ""
most_popular = 10000


data = pd.read_csv(r"C:\Users\robin\Desktop\Graduation_Project\cikm\train-item-views.csv",
                   sep=";", header=0, usecols=[0, 2, 3, 4], dtype={0: np.int32, 1: np.int64, 2: np.int32, 3: str})
# data.columns = ["sessionId", "TimeStr", "itemId"]
data["Time"] = data["eventdate"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").timestamp()) #This is not UTC. It does not really matter.
del(data["eventdate"])
data.rename(columns={"sessionId": "SessionId", "itemId": "ItemId"}, inplace=True)
print(data.head())

session_lengths = data.groupby("SessionId").size()
data = data[np.in1d(data.SessionId, session_lengths[(20 >= session_lengths) & (session_lengths > 1)].index)]
item_supports = data.groupby("ItemId").size()
item_supports = item_supports.sort_values(ascending=False).iloc[:most_popular]
data = data[np.in1d(data.ItemId, item_supports[item_supports >= 5].index)]
session_lengths = data.groupby("SessionId").size()
data = data[np.in1d(data.SessionId, session_lengths[(20 >= session_lengths) & (session_lengths > 1)].index)]

tmax = data.Time.max()
session_max_times = data.groupby("SessionId").Time.max()
session_train = session_max_times[session_max_times < tmax-86400*7].index
session_test = session_max_times[session_max_times > tmax-86400*7].index

train = data[np.in1d(data.SessionId, session_train)]
trlength = train.groupby("SessionId").size()
train = train[np.in1d(train.SessionId, trlength[(20 >= trlength) & (trlength >= 2)].index)]
test = data[np.in1d(data.SessionId, session_test)]
test = test[np.in1d(test.ItemId, train.ItemId)]
tslength = test.groupby("SessionId").size()
test = test[np.in1d(test.SessionId, tslength[(20 >= tslength) & (tslength >= 2)].index)]
print("Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}"
      .format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
train.to_csv(PATH_TO_PROCESSED_DATA + "cikm16_train.txt", sep="\t", index=False)
print("Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}"
      .format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
test.to_csv(PATH_TO_PROCESSED_DATA + "cikm16_test.txt", sep="\t", index=False)
print(train.head())


"""
10000:
Full train set
	Events: 543049
	Sessions: 135854
	Items: 10000
Test set
	Events: 46273
	Sessions: 11553
	Items: 7597
	
20000:
Full train set
	Events: 704604
	Sessions: 162887
	Items: 19998
Test set
	Events: 60069
	Sessions: 13872
	Items: 12960
15000:（也不行了）
Full train set
	Events: 637106
	Sessions: 152186
	Items: 15000
Test set
	Events: 54512
	Sessions: 12998
	Items: 10546
12500:
Full train set
	Events: 594251
	Sessions: 144939
	Items: 12500
Test set
	Events: 50880
	Sessions: 12379
	Items: 9165
"""
