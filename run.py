"""
# -*- coding: utf-8 -*-
# @FileName: Run.py
# @Author  : Robin
# @Time    : 2020/2/18 20:27
"""
import sys

sys.path.append("./")
from torch import optim
from TrainerUtils.CumulativeTrainer import *
import torch.backends.cudnn as cudnn
import argparse
from BlurRepeatNet.Model import *
import codecs
import numpy as np
import random

from BlurRepeatNet.Dataset import RepeatNetDataset, collate_fn


def get_millisecond():
    return time.time() * 1000


# initialize seed for random
def init_seed(seed=None):
    if seed is None:
        seed = int(get_millisecond() // 1000)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


base_output_path = r"./Output/RepeatNet/"
base_data_path = r"./Datasets/"
dir_path = os.path.dirname(os.path.realpath(__file__))

epochs = 100
embedding_size = 128
hidden_size = 128
dataset2item_vocab_size = {"cikm16": 10000+1, "rsc15": 7682+1, "demo": 10+1}


def train(args):
    batch_size = 512

    train_dataset = RepeatNetDataset(base_data_path + args.dataset + "/" + args.dataset + "_train_rn.txt", args)
    # train_dataset = RepeatNetDataset(base_data_path + args.dataset + "/demo.train", args)

    model = BlurRepeatNet(embedding_size, hidden_size, dataset2item_vocab_size[args.dataset])
    init_params(model)

    trainer = CumulativeTrainer(model, None, None, args.local_rank, 4)
    model_optimizer = optim.Adam(model.parameters())

    # 没必要保留前20个epochs的模型
    for i in range(epochs):
        trainer.train_epoch("train", train_dataset, collate_fn, batch_size, i, model_optimizer)
        if i >= 0:
            trainer.serialize(i, output_path=base_output_path)


def infer(args):
    batch_size = 512

    test_dataset = RepeatNetDataset(base_data_path + args.dataset + "/" + args.dataset + "_test_rn.txt", args)
    # test_dataset = RepeatNetDataset(base_data_path + args.dataset + "/demo.test", args)

    # 节省时间
    for i in range(20, epochs):
        print("epoch", i)
        file = base_output_path + "model/" + str(i) + ".pkl"

        model = BlurRepeatNet(embedding_size, hidden_size, dataset2item_vocab_size[args.dataset])
        if os.path.exists(file):
            model.load_state_dict(torch.load(file, map_location='cpu'))
        trainer = CumulativeTrainer(model, None, None, args.local_rank, 4)

        rs = trainer.predict("infer", test_dataset, collate_fn, batch_size, i, base_output_path)
        file = codecs.open(base_output_path + "result/" + args.dataset + "/" + str(i) + "." + str(args.local_rank) + ".test", mode="w",
                           encoding="utf-8")
        for data, output in rs:
            scores, index = output
            label = data["item_tgt"]
            for j in range(label.size(0)):
                file.write("[" + ",".join([str(id) for id in index[j, :50].tolist()]) + "]|[" + ",".join(
                    [str(id) for id in label[j].tolist()]) + "]" + os.linesep)  # score ranking?
        file.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定第一块gpu
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--mode", type=str, default="infer", choices="train,infer")
    parser.add_argument("--dataset", type=str, default="cikm16", choices="rsc15,cikm16,demo")
    args = parser.parse_args()

    init_seed(23)

    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)
    else:
        raise Exception("no support")
