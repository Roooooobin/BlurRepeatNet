import matplotlib.pyplot as plt
import numpy as np
from Utils.decorator import time_trace


@time_trace
def present(evaluation: map, evaluation_metrics: str, cut_off):
    gap_width = 0.2
    plt.figure(figsize=(12, 8), dpi=80)
    # 分别绘制各模型的评测结果
    x_pop = [0]
    x_spop = [i + gap_width for i in x_pop]
    x_itemknn = [i + gap_width for i in x_spop]
    x_bpr = [i + gap_width for i in x_itemknn]
    x_gru = [i + gap_width for i in x_bpr]
    x_rn = [i + gap_width for i in x_gru]
    x_brn = [i + gap_width for i in x_rn]
    plt.bar(x_pop, evaluation["Pop"], width=gap_width, label="Pop", color="#000080")
    plt.bar(x_spop, evaluation["S-Pop"], width=gap_width, label="S-Pop", color="#FF0000")
    plt.bar(x_itemknn, evaluation["Item-KNN"], width=gap_width, label="Item-KNN", color="#1E90FF")
    plt.bar(x_bpr, evaluation["BPR"], width=gap_width, label="BPR", color="#FFD700")
    plt.bar(x_gru, evaluation["GRU4REC"], width=gap_width, label="GRU4REC", color="#123456")
    plt.bar(x_rn, evaluation["RN"], width=gap_width, label="RN", color="#008000")
    plt.bar(x_brn, evaluation["BRN"], width=gap_width, label="BRN", color="#0000CD")

    plt.legend(loc="upper center", fontsize="x-large", shadow=True, ncol=4)

    x_tick = [0.6]
    x_tick_label = ["test"]

    y_max = -1
    for key, value in evaluation.items():
        y_max = max(value, y_max)
    y_tick_diff = 0.1 if evaluation_metrics.find("Recall") != -1 else 0.05   # TODO: 根据具体数值调整到最佳的显示效果
    y_tick = [i + y_tick_diff for i in np.arange(0, y_max+y_tick_diff, y_tick_diff)]

    plt.xticks(x_tick, x_tick_label, fontsize=15)
    plt.yticks(y_tick, fontsize=15)
    plt.ylabel("{}@{}".format(evaluation_metrics, cut_off), fontsize=20)

    plt.savefig("{}@{}.png".format(evaluation_metrics, cut_off))
    plt.show()


if __name__ == "__main__":
    # 20
    method_to_recall20_cikm16 = {
        'Pop': 0.014660138248847926,    # 1.47
        'S-Pop': 0.2444700460829493,
        'Item-KNN': 0.4075748847926267,
        'BPR': 0.07019009216589862,
        'GRU4REC': 0.3491647465437788
    }

    method_to_mrr20_cikm16 = {
        'Pop': 0.0037266306298818566,
        'S-Pop': 0.18480570337354013,
        'Item-KNN': 0.1409033008087476,
        'BPR': 0.02560850456040952,
        'GRU4REC': 0.09089602171179709
    }

    method_to_recall20_rsc15 = {
        'Pop': 0.06047456693184557,
        'S-Pop': 0.2994799738264972,
        'Item-KNN': 0.5357302751661673,
        'BPR': 0.09012639046733478,
        'GRU4REC': 0.59563256125436234,
    }

    method_to_mrr20_rsc15 = {
        'Pop': 0.013010801021649553,
        'S-Pop': 0.19508645260549756,
        'Item-KNN': 0.23332804836988577,
        'BPR': 0.02018540037089487,
        'GRU4REC': 0.24768655071145699
    }

    # 10
    method_to_recall10_cikm16 = {
        'Pop': 0.009360599078341013,
        'S-Pop': 0.240610599078341,
        'Item-KNN': 0.298934331797235,
        'BPR': 0.04567972350230415,
        'GRU4REC': 0.2154377880184332
    }

    method_to_mrr10_cikm16 = {
        'Pop': 0.003377004699729356,
        'S-Pop': 0.18456663512178906,
        'Item-KNN': 0.13269187513715272,
        'BPR': 0.023842920232609144,
        'GRU4REC': 0.08105798771121354
    }

    method_to_recall10_rsc15 = {
        'Pop': 0.04180872679684541,
        'S-Pop': 0.28401694389916315,
        'Item-KNN': 0.43613320935358335,
        'BPR': 0.04776664255949306,
        'GRU4REC': 0.47204324532245124
    }

    method_to_mrr10_rsc15 = {
        'Pop': 0.01173732364454534,
        'S-Pop': 0.1941674169409455,
        'Item-KNN': 0.22568798374924554,
        'BPR': 0.01720404891186992,
        'GRU4REC': 0.23710652818564232
    }

    method_to_recall20_cikm16["BRN"], method_to_mrr20_cikm16["BRN"] = 0.641620253164557, 0.24181675802501504
    method_to_recall10_cikm16["BRN"], method_to_mrr10_cikm16["BRN"] = 0.5215189873417722, 0.2333882459312847
    method_to_recall20_rsc15["BRN"], method_to_mrr20_rsc15["BRN"] = 0.6797738259668509, 0.29244225321151623
    method_to_recall10_rsc15["BRN"], method_to_mrr10_rsc15["BRN"] = 0.5646581491712708, 0.28439468026505804

    method_to_recall20_cikm16["RN"], method_to_mrr20_cikm16["RN"] = 0.6425316455696203, 0.2409104167081255
    method_to_recall10_cikm16["RN"], method_to_mrr10_cikm16["RN"] = 0.5210632911392405, 0.2322946554149089
    method_to_recall20_rsc15["RN"], method_to_mrr20_rsc15["RN"] = 0.6825362569060773, 0.292497645207035
    method_to_recall10_rsc15["RN"], method_to_mrr10_rsc15["RN"] = 0.5625, 0.28426743510479263

    # baselines
    # method_to_recall20_cikm16, method_to_recall10_cikm16, method_to_mrr20_cikm16, method_to_mrr10_cikm16 = main("cikm16")
    # method_to_recall20_rsc15, method_to_recall10_rsc15, method_to_mrr20_rsc15, method_to_mrr10_rsc15 = main("rsc15")
    # RepeatNet, BlurRepeatNet
    # method_to_recall20_cikm16["BRN"], method_to_mrr20_cikm16["BRN"] = evaluate_rn("BlurRepeatNet", "cikm16", 20)
    # method_to_recall10_cikm16["BRN"], method_to_mrr10_cikm16["BRN"] = evaluate_rn("BlurRepeatNet", "cikm16", 10)
    # method_to_recall20_rsc15["BRN"], method_to_mrr20_rsc15["BRN"] = evaluate_rn("BlurRepeatNet", "rsc15", 20)
    # method_to_recall10_rsc15["BRN"], method_to_mrr10_rsc15["BRN"] = evaluate_rn("BlurRepeatNet", "rsc15", 10)
    #
    # method_to_recall20_cikm16["RN"], method_to_mrr20_cikm16["RN"] = evaluate_rn("RepeatNet", "cikm16", 20)
    # method_to_recall10_cikm16["RN"], method_to_mrr10_cikm16["RN"] = evaluate_rn("RepeatNet", "cikm16", 10)
    # method_to_recall20_rsc15["RN"], method_to_mrr20_rsc15["RN"] = evaluate_rn("RepeatNet", "rsc15", 20)
    # method_to_recall10_rsc15["RN"], method_to_mrr10_rsc15["RN"] = evaluate_rn("RepeatNet", "rsc15", 10)

    present(method_to_recall10_cikm16, "CIKM-Recall", 10)
    present(method_to_recall20_cikm16, "CIKM-Recall", 20)
    present(method_to_mrr20_cikm16, "CIKM-MRR", 20)
    present(method_to_mrr10_cikm16, "CIKM-MRR", 10)

    present(method_to_recall10_rsc15, "RSC64-Recall", 10)
    present(method_to_recall20_rsc15, "RSC64-Recall", 20)
    present(method_to_mrr10_rsc15, "RSC64-MRR", 10)
    present(method_to_mrr20_rsc15, "RSC64-MRR", 20)

    """
    evaluation records
    cikm16(recall20,10; mrr20,10)
    {'Pop': 0.014660138248847926, 'S-Pop': 0.2444700460829493, 'Item-KNN': 0.4075748847926267, 'BPR': 0.07019009216589862}
    {'Pop': 0.009360599078341013, 'S-Pop': 0.240610599078341, 'Item-KNN': 0.298934331797235, 'BPR': 0.04567972350230415}
    {'Pop': 0.0037266306298818566, 'S-Pop': 0.18480570337354013, 'Item-KNN': 0.1409033008087476, 'BPR': 0.02560850456040952}
    {'Pop': 0.003377004699729356, 'S-Pop': 0.18456663512178906, 'Item-KNN': 0.13269187513715272, 'BPR': 0.023842920232609144}
    
    rsc15(recall20,10; mrr20,10)
    {'Pop': 0.06047456693184557, 'S-Pop': 0.2994799738264972, 'Item-KNN': 0.5357302751661673, 'BPR': 0.09012639046733478}
    {'Pop': 0.04180872679684541, 'S-Pop': 0.28401694389916315, 'Item-KNN': 0.43613320935358335, 'BPR': 0.04776664255949306}
    {'Pop': 0.013010801021649553, 'S-Pop': 0.19508645260549756, 'Item-KNN': 0.23332804836988577, 'BPR': 0.02018540037089487}
    {'Pop': 0.01173732364454534, 'S-Pop': 0.1941674169409455, 'Item-KNN': 0.22568798374924554, 'BPR': 0.01720404891186992}
    
    BlurRepeatNet
    cikm16: recall20, mrr20; recall10, mrr10
    0.641620253164557 0.24181675802501504
    0.5215189873417722 0.2333882459312847
    rsc15:
    0.6797738259668509 0.29244225321151623
    0.5646581491712708 0.28439468026505804
    
    RepeatNet
    cikm16:
    0.6425316455696203 0.2409104167081255
    0.5210632911392405 0.2322946554149089
    rsc15:
    0.6825362569060773 0.292497645207035
    0.5625 0.28426743510479263
    """