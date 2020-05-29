import pandas as pd
import numpy as np
import os
from Baselines.baselines import Pop, SessionPop, ItemKNN, BPR
from Evaluations.evaluation_baseline import evaluate

cikm16_train = r"../Datasets/cikm16/cikm16_train.txt"
cikm16_test = r"../Datasets/cikm16/cikm16_test.txt"
rsc15_train_64 = r"../Datasets/rsc15/rsc15_64_train.txt"
rsc15_test_64 = r"../Datasets/rsc15/rsc15_64_test.txt"


def load_data(dataset):
    train_data_path = {"cikm16": cikm16_train, "rsc15": rsc15_train_64}
    test_data_path = {"cikm16": cikm16_test, "rsc15": rsc15_test_64}

    train_data = pd.read_table(train_data_path[dataset])
    test_data = pd.read_table(test_data_path[dataset])
    recommended_list = test_data.ItemId.unique()[np.in1d(test_data.ItemId.unique(), train_data.ItemId.unique())]
    return train_data, test_data, recommended_list


def main(dataset):
    train_data, test_data, recommended_list = load_data(dataset)

    # model fitting
    pop_model = Pop()
    pop_model.fit(train_data)
    spop_model = SessionPop()
    spop_model.fit(train_data)
    itemknn_model = ItemKNN()
    itemknn_model.fit(train_data)
    bpr_model = BPR()
    bpr_model.fit(train_data)

    method_to_recall_20 = {}
    method_to_recall_10 = {}
    method_to_mrr_20 = {}
    method_to_mrr_10 = {}
    
    pop_recall_20, pop_mrr_20 = evaluate(pop_model, recommended_list, test_data)
    spop_recall_20, spop_mrr_20 = evaluate(spop_model, recommended_list, test_data)
    itemknn_recall_20, itemknn_mrr_20 = evaluate(itemknn_model, recommended_list, test_data)
    bpr_recall_20, bpr_mrr_20 = evaluate(bpr_model, recommended_list, test_data)
    method_to_recall_20["Pop"] = pop_recall_20
    method_to_recall_20["S-Pop"] = spop_recall_20
    method_to_recall_20["Item-KNN"] = itemknn_recall_20
    method_to_recall_20["BPR"] = bpr_recall_20
    method_to_mrr_20["Pop"] = pop_mrr_20
    method_to_mrr_20["S-Pop"] = spop_mrr_20
    method_to_mrr_20["Item-KNN"] = itemknn_mrr_20
    method_to_mrr_20["BPR"] = bpr_mrr_20

    pop_recall_10, pop_mrr_10 = evaluate(pop_model, recommended_list, test_data, cut_off=10)
    spop_recall_10, spop_mrr_10 = evaluate(spop_model, recommended_list, test_data, cut_off=10)
    itemknn_recall_10, itemknn_mrr_10 = evaluate(itemknn_model, recommended_list, test_data, cut_off=10)
    bpr_recall_10, bpr_mrr_10 = evaluate(bpr_model, recommended_list, test_data, cut_off=10)
    method_to_recall_10["Pop"] = pop_recall_10
    method_to_recall_10["S-Pop"] = spop_recall_10
    method_to_recall_10["Item-KNN"] = itemknn_recall_10
    method_to_recall_10["BPR"] = bpr_recall_10
    method_to_mrr_10["Pop"] = pop_mrr_10
    method_to_mrr_10["S-Pop"] = spop_mrr_10
    method_to_mrr_10["Item-KNN"] = itemknn_mrr_10
    method_to_mrr_10["BPR"] = bpr_mrr_10

    print(method_to_recall_20)
    print(method_to_recall_10)
    print(method_to_mrr_20)
    print(method_to_mrr_10)
    return method_to_recall_20, method_to_recall_10, method_to_mrr_20, method_to_mrr_10


if __name__ == "__main__":
    main("rsc15")
    main("cikm16")
