from torch.utils.data import Dataset
import csv
import codecs
import torch
import pickle
from BlurRepeatNet.Blur import calc_similarity_interest_shift
from BlurRepeatNet.Blur import calc_similarity_behavior_shift


class RepeatNetDataset(Dataset):
    def __init__(self, sample_file, args):
        super(RepeatNetDataset, self).__init__()

        self.sample_file = sample_file

        self.item_atts = dict()
        self.samples = []
        self.dataset = args.dataset
        self.mode = args.mode
        # if self.mode == "train":
        #     self.similarity = calc_similarity_interest_shift(self.dataset)  # without loading
        if self.mode == "train":
            self.similarity = calc_similarity_behavior_shift(self.sample_file)  # without loading
        self.load()

    def load(self):
        clean = lambda l: [int(x) for x in l.strip("[]").split(",")]
        len_ne0 = lambda l: len([x for x in l if x != 0])

        idx = 0
        # [81766, 31331, 32118, ..., 0]|[9654]  filled with 0
        with codecs.open(self.sample_file, encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter="|")
            sample_added = []
            for row in csv_reader:
                idx += 1
                # [tensor([3]), tensor([81766, 31331, 32118]), tensor([9654])]
                self.samples.append([torch.tensor([idx]), torch.tensor(clean(row[0])), torch.tensor(clean(row[1]))])
                if self.mode == "train":
                    if len_ne0(clean(row[0])) >= 9:
                        if clean(row[1])[0] not in self.similarity:
                            continue
                        idx += 1
                        tgt_new = list(self.similarity[clean(row[1])[0]].keys())[0]
                        score = list(self.similarity[clean(row[1])[0]].values())[0]
                        sample_added.append([[torch.tensor([idx]), torch.tensor(clean(row[0])), torch.tensor([tgt_new])], score])
            if self.mode == "train":
                sample_added_sorted = list(sorted(sample_added, key=lambda l: l[1], reverse=True))
                for x in sample_added_sorted[:len(sample_added_sorted)//10]:
                    self.samples.append(x[0])

        self.len = len(self.samples)
        print("data size: ", self.len)

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return self.len


def collate_fn(data):
    idx, item_seq, item_tgt = zip(*data)

    return {
            "id": torch.cat(idx),
            "item_seq": torch.stack(item_seq),
            "item_tgt": torch.stack(item_tgt)
            }
