# try to transform data to the standard format for BlurRepeatNet
import os
import numpy as np
import pandas as pd
import codecs
import pickle

cikm16 = r"../Datasets/cikm16/cikm16_"
rsc15_64 = r"../Datasets/rsc15/rsc15_64_"

session_max_len = 20


def process_data(base_file_path: str):
    """
    first step:

    each session is transformed to a single item-list: [1, 2, 3, 4](ItemId)

    second step:
    item-list transform to the following format
    [1]|[2]
    [1, 2]|[3]
    [1, 2, 3]|[4]
    """
    train_path = base_file_path + "train.txt"
    train_data = pd.read_table(train_path)
    array = train_data.values
    pre_session_id = array[0][0]
    session_items = []
    items = []
    id_remapping = {}
    idx = 1
    for arr in array:
        cur_session_id = arr[0]
        item_id = str(int(arr[1]))
        if cur_session_id != pre_session_id:
            session_items.append(items)
            items = []
        if item_id not in id_remapping:
            items.append(str(idx))
            id_remapping[item_id] = str(idx)
            idx += 1
        else:
            items.append(id_remapping[item_id])
        pre_session_id = cur_session_id

    file_name = train_path[train_path.find("/", train_path.find("/") + 1) + 1:-4]
    folder_name = base_file_path[:train_path.index(file_name)]
    file = folder_name + file_name
    f_mid = codecs.open("{}_.txt".format(file), encoding="utf-8", mode="w")
    for session in session_items:
        f_mid.write(", ".join(session) + os.linesep)
    f = codecs.open("{}_rn.txt".format(file), encoding="utf-8", mode="w")
    with codecs.open("{}_.txt".format(file), encoding="utf-8") as f_mid:
        for line in f_mid:
            lines = line.strip("\n").strip("\r").split(", ")
            for i in range(len(lines) - 1):
                items = lines[0: i+1]
                items.extend(["0"] * (session_max_len - len(items)))
                f.write("[" + ", ".join(items) + "]" + "|[" + lines[i+1] + "]" + os.linesep)

    print(len(id_remapping))
    os.remove("{}_.txt".format(file))
    f.close()

    test_path = base_file_path + "test.txt"
    test_data = pd.read_table(test_path)
    array = test_data.values
    pre_session_id = array[0][0]
    session_items = []
    items = []
    for arr in array:
        cur_session_id = arr[0]
        item_id = str(int(arr[1]))
        if pre_session_id == cur_session_id:
            if item_id not in id_remapping:
                items.append(str(idx))
                id_remapping[item_id] = str(idx)
                idx += 1
            else:
                items.append(id_remapping[item_id])
        else:
            session_items.append(items)
            items = []
        pre_session_id = cur_session_id

    file_name = test_path[test_path.find("/", test_path.find("/") + 1) + 1:-4]
    folder_name = test_path[:test_path.index(file_name)]
    file = folder_name + file_name
    f_mid = codecs.open("{}_.txt".format(file), encoding="utf-8", mode="w")
    for session in session_items:
        f_mid.write(", ".join(session) + os.linesep)
    f = codecs.open("{}_rn.txt".format(file), encoding="utf-8", mode="w")
    with codecs.open("{}_.txt".format(file), encoding="utf-8") as f_mid:
        for line in f_mid:
            lines = line.strip("\n").strip("\r").split(", ")
            for i in range(len(lines) - 1):
                items = lines[0: i+1]
                items.extend(["0"] * (session_max_len - len(items)))
                f.write("[" + ", ".join(items) + "]" + "|[" + lines[i+1] + "]" + os.linesep)

    print(len(id_remapping))
    file_pickle = open("{}_item_mapping".format(file), "wb")
    pickle.dump(id_remapping, file_pickle)
    os.remove("{}_.txt".format(file))
    f.close()


if __name__ == "__main__":
    process_data(cikm16)
    process_data(rsc15_64)
