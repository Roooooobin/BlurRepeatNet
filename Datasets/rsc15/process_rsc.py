"""
Created on Fri Jun 25 16:20:12 2015
@author: Balázs Hidasi

Modified on Fri Jan 31 15:54:23 2020
@author: Robin
"""

import numpy as np
import pandas as pd
import datetime as dt

PATH_TO_ORIGINAL_DATA = ""
PATH_TO_PROCESSED_DATA = ""


def generate_rsc_data(pro):
    data = pd.read_csv(r"C:\Users\robin\Desktop\documents\Graduation_Project\yoochoose-data\yoochoose-clicks.dat",
                       sep=",", header=None, usecols=[0, 1, 2], dtype={0: np.int32, 1: str, 2: np.int64})
    data.columns = ["SessionId", "TimeStr", "ItemId"]
    data["Time"] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())
    del (data["TimeStr"])

    print(dt.datetime.now())

    data.sort_values(["SessionId", "Time"], inplace=True)
    print("sort finish")
    data = data[-len(data) // pro:]
    i = 0
    session_data = data["SessionId"].values
    for i in range(len(data)):
        if session_data[i] != session_data[i + 1]:
            break
    data = data[i + 1:]

    print(dt.datetime.now())

    session_lengths = data.groupby("SessionId").size()
    data = data[np.in1d(data.SessionId, session_lengths[(20 >= session_lengths) & (session_lengths > 1)].index)]
    item_supports = data.groupby("ItemId").size()
    data = data[np.in1d(data.ItemId, item_supports[item_supports >= 5].index)]
    session_lengths = data.groupby("SessionId").size()
    data = data[np.in1d(data.SessionId, session_lengths[(20 >= session_lengths) & (session_lengths > 1)].index)]

    print(dt.datetime.now())

    tmax = data.Time.max()
    session_max_times = data.groupby("SessionId").Time.max()
    session_train = session_max_times[session_max_times < tmax - 86400].index
    session_test = session_max_times[session_max_times > tmax - 86400].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby("SessionId").size()
    test = test[np.in1d(test.SessionId, tslength[(20 >= tslength) & (tslength >= 2)].index)]
    print("Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}"
          .format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(PATH_TO_PROCESSED_DATA + "rsc15_train_{}.txt".format(str(pro)), sep="\t", index=False)
    print("Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}"
          .format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(PATH_TO_PROCESSED_DATA + "rsc15_test_{}.txt".format(str(pro)), sep="\t", index=False)


if __name__ == "__main__":
    # generate_rsc_data(4)
    generate_rsc_data(64)
