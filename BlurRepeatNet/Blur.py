import codecs
import csv

import pandas as pd
import pickle
import os
from collections import defaultdict
from math import sqrt
from Utils.decorator import time_trace

dataset2mapping_file = {
    "rsc15": os.getcwd() + "../Datasets/rsc15/rsc15_test_item_mapping",
    "cikm16": os.getcwd() + "../Datasets/cikm16/cikm16_test_item_mapping"
}
dataset2dataset_file = {
    "rsc15": os.getcwd() + "../Datasets/rsc15/rsc15_64_train.txt",
    "cikm16": os.getcwd() + "../Datasets/cikm16/cikm16_train.txt"
}

"""
会话内时间无关的物品相似度计算方法：
适用于时间偏移型数据
由于会话内部不包含时间推移，可以认为两个物品出现在相同会话的比例越大，越相似
公式为sim(i, j) = c(i, j) / sqrt(c(i) * c(j))，c(i, j)表示物品i和物品j出现在同一个会话的次数，c(i), c(j)分别表示物品i、j所在的会话数
"""
@time_trace
def calc_similarity_interest_shift(dataset):
    file_mapping = open(dataset2mapping_file[dataset], "rb")
    item_mapping = pickle.load(file_mapping)
    # 由于处理数据集时，ItemId重新做了一步映射，所以这里载入并映射
    item_mapping_int = {int(key): int(val) for key, val in item_mapping.items()}
    data = pd.read_table(dataset2dataset_file[dataset])
    data["ItemId"] = data["ItemId"].map(item_mapping_int)  # 此时得到的data是映射后的
    # item2session: 记录的是某物品item对应的所在会话编号，len(set)表示出现的次数
    item2session = defaultdict(set)
    item_ids = set()
    # 一条会话中的各个物品的会话集set加入该会话id
    for idx, row in data.iterrows():
        cur_session_id = int(row["SessionId"])
        item_id = int(row["ItemId"])
        item_ids.add(item_id)
        item2session[item_id].add(cur_session_id)

    item_ids = list(item_ids)

    # 通过两个物品set的与运算后的set大小计算出两个物品出现在同一会话的次数
    # 得到的是每个item对应其他所有item的相似度值
    sim = {
        a: {
            b: len(item2session[a] & item2session[b]) / sqrt(len(item2session[a]) * len(item2session[b]))
            for b in item_ids if a != b
        } for a in item_ids
    }

    # 计算完后按相似度值降序排序
    sim_sorted = {key: dict(sorted(val_dict.items(), key=lambda d: d[1], reverse=True))
                  for key, val_dict in sim.items()}

    # 可以通过pickle保存结果
    # f_sim = open(similarity_file, "wb")
    # pickle.dump(item2item2similarity_sorted, f_sim)

    return sim_sorted


"""
会话内时间相关的物品相似度计算方法：
适用于行为偏移型数据
因为会话内部存在时间推移，一条会话可能是一个随时间变化的行为链，那么两个物品出现在相同会话多并不一定就更相似
将每条会话的最后一个物品之前的物品作为其特征向量，使用向量间的相似度计算物品之间的相似度
"""
@time_trace
def calc_similarity_behavior_shift(dataset_file):
    # 规范数据集的会话格式
    clean = lambda l: [int(x) for x in l.strip("[]").split(",")]

    item_ids = set()
    sessions = []
    # 需要录入整个会话集
    with codecs.open(dataset_file, encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter="|")
        # row[0] is item_seq, row[1] is item_tgt
        for row in csv_reader:
            item_seq = [x for x in clean(row[0]) if x != 0]
            item_tgt = clean(row[1])[0]
            item_ids.add(item_tgt)
            # 过滤会话长度小于9的会话
            if len(item_seq) >= 9:
                sessions.append((item_tgt, item_seq))

    # calc similarity，得到的是物品对于其他所有物品的相似度分数
    sim = {}
    # import random
    # sessions = list(random.sample(sessions, len(sessions)//4))
    # item_seq作为item_tgt的特征向量，按照带有位置权重的向量间相似度计算方式计算两个物品间的相似度
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            item1 = sessions[i][0]
            item2 = sessions[j][0]
            if item1 == item2:
                continue
            sim.setdefault(item1, {})
            sim[item1].setdefault(item2, 0)
            # 一个物品的特征向量可能不止一个，选取其中最大的相似度作为衡量
            sim[item1][item2] = max(sim[item1][item2], calc_similar_score(sessions[i][1], sessions[j][1])
                                    / sqrt(len(sessions[i][1]) * len(sessions[j][1])))
            sim.setdefault(item2, {})
            sim[item2].setdefault(item1, 0)
            sim[item2][item1] = sim[item1][item2]
    # 计算完毕，降序排序
    sim_sorted = {key: dict(sorted(val_dict.items(), key=lambda d: d[1], reverse=True))
                  for key, val_dict in sim.items()}

    return sim_sorted


"""
计算两个特征向量之间的相似度，考虑位置权重
位置比例相似获得更高权重，从而在相同比例位置命中更多的两个特征向量会更相似
"""
def calc_similar_score(vi, vj):
    ni = len(vi)
    nj = len(vj)
    score = 0.0
    for i in range(ni):
        for j in range(nj):
            score += 1.0
            if vi[i] == vj[j]:
                # i / ni表示物品vi[i]的位置比例
                if abs(i / ni - j / nj) <= (1 / max((ni, nj))):
                    score += 1.0

    return score


# @time_trace
# def calc_similarity_3(dataset_file):
#     clean = lambda l: [int(x) for x in l.strip("[]").split(",")]
#
#     item2items = {}
#     item_ids = set()
#     sessions = []
#     with codecs.open(dataset_file, encoding="utf-8") as f:
#         csv_reader = csv.reader(f, delimiter="|")
#         # row[0] is item_seq, row[1] is item_tgt
#         for row in csv_reader:
#             item_seq = [x for x in clean(row[0]) if x != 0]
#             item_tgt = clean(row[1])[0]
#             item_ids.add(item_tgt)
#             if len(item_seq) >= 9:
#                 sessions.append((item_tgt, set(item_seq)))
#
#     # calc similarity
#     item2item2similarity = {}
#     print(len(sessions))
#     print(len(item_ids))
#     for i in range(len(sessions)):
#         for j in range(i+1, len(sessions)):
#             item1 = sessions[i][0]
#             item2 = sessions[j][0]
#             if item1 == item2:
#                 continue
#             item2item2similarity.setdefault(item1, {})
#             item2item2similarity[item1].setdefault(item2, 0)
#             item2item2similarity[item1][item2] = max(item2item2similarity[item1][item2],
#                                                      len(sessions[i][1] & sessions[j][1]) / sqrt(
#                                                          len(sessions[i][1]) * len(sessions[j][1])))
#             item2item2similarity.setdefault(item2, {})
#             item2item2similarity[item2].setdefault(item1, 0)
#             item2item2similarity[item2][item1] = item2item2similarity[item1][item2]
#     item2item2similarity_sorted = {key: dict(sorted(val_dict.items(), key=lambda d: d[1], reverse=True))
#                                    for key, val_dict in item2item2similarity.items()}
#
#     return item2item2similarity_sorted


if __name__ == "__main__":
    sim1 = calc_similarity_interest_shift(r"")
    sim2 = calc_similarity_behavior_shift(r"")
