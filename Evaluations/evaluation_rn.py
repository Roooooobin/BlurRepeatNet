import csv
import os
import codecs

clean = lambda l: [int(x) for x in l.strip("[]").split(",")]


def evaluate_rn(model, dataset, cut_off):
    path = "../output/{}/result/{}".format(model, dataset)
    recall_max = 0.0
    recall_sum = 0.0
    mrr_max = 0.0
    mrr_sum = 0.0
    cnt_sum = 0.0
    for filename in os.listdir(path):
        full_filename = os.path.join(path, filename)
        if os.path.isfile(full_filename):
            with codecs.open(full_filename, encoding="utf-8") as f:
                csv_reader = csv.reader(f, delimiter="|")
                recall = 0.0
                mrr = 0.0
                cnt = 0.0
                for row in csv_reader:
                    prediction_ids = clean(row[0])
                    target_id = clean(row[1])[0]
                    cnt_sum += 1
                    cnt += 1
                    for i in range(0, cut_off):
                        if prediction_ids[i] == target_id:
                            recall += 1
                            mrr += 1.0 / (i+1)
                recall_max = max(recall_max, recall / cnt)
                mrr_max = max(mrr_max, mrr / cnt)
                recall_sum += recall
                mrr_sum += mrr
                # print(recall / cnt, mrr / cnt)

    print(recall_max, mrr_max)
    return recall_max, mrr_max


if __name__ == "__main__":
    evaluate_rn("RepeatNet", "rsc15", 20)
    evaluate_rn("RepeatNet", "rsc15", 10)
    evaluate_rn("RepeatNet", "cikm16", 20)
    evaluate_rn("RepeatNet", "cikm16", 10)
