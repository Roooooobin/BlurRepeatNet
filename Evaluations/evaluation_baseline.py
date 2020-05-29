import matplotlib.pyplot as plt
import numpy as np

from Utils.decorator import time_trace


@time_trace
def evaluate(predict_model, recommend_list, test_data, items_to_compare=None, cut_off=20,
             session_key="SessionId", item_key="ItemId", time_key="Time"):
    test_data.sort_values([session_key, time_key], inplace=True)    # sort by session_id and timestamp(ascending)
    items_recommended = recommend_list
    evaluation_point_count = 0
    pre_item_id, pre_session_id = -1, -1
    recall, mrr = 0.0, 0.0
    for i in range(len(test_data)):
        session_id = test_data[session_key].values[i]
        item_id = test_data[item_key].values[i]
        if pre_session_id != session_id:
            pre_session_id = session_id
        else:
            if items_to_compare:
                if np.in1d(item_id, items_to_compare):
                    items_recommended = items_to_compare
                else:
                    items_recommended = np.hstack(([item_id], items_to_compare))
            predictions = predict_model.predict(session_id, pre_item_id, items_recommended)
            predictions[np.isnan(predictions)] = 0
            predictions += 1e-8 * np.random.rand(len(predictions))
            rank = (predictions > predictions[item_id]).sum() + 1
            if rank < cut_off:
                recall += 1
                mrr += 1.0 / rank
            evaluation_point_count += 1
        pre_item_id = item_id
    return recall / evaluation_point_count, mrr / evaluation_point_count


@time_trace
def present(evaluation: map, evaluation_metrics: str, cut_off):
    gap_width = 0.2
    plt.figure(figsize=(12, 8), dpi=80)
    # 分别绘制各模型的评测结果
    x_pop = [0]
    x_spop = [i + gap_width for i in x_pop]
    x_itemknn = [i + gap_width for i in x_spop]
    x_bpr = [i + gap_width for i in x_itemknn]
    x_rn = [i + gap_width for i in x_bpr]
    x_brn = [i + gap_width for i in x_rn]
    plt.bar(x_pop, evaluation["Pop"], width=gap_width, label="Pop", color="#000080")
    plt.bar(x_spop, evaluation["S-Pop"], width=gap_width, label="S-Pop", color="#FF0000")
    plt.bar(x_itemknn, evaluation["Item-KNN"], width=gap_width, label="Item-KNN", color="#1E90FF")
    plt.bar(x_bpr, evaluation["BPR"], width=gap_width, label="BPR", color="#FFD700")
    plt.bar(x_rn, evaluation["RN"], width=gap_width, label="RN", color="#008000")
    plt.bar(x_brn, evaluation["BRN"], width=gap_width, label="BRN", color="#0000CD")

    plt.legend(loc="upper center", fontsize="xx-large", shadow=True, ncol=6)

    x_tick = [0.5]
    x_tick_label = ["test"]

    y_max = -1
    for key, value in evaluation.items():
        y_max = max(max(value), y_max)
    y_tick_diff = 0.1 if evaluation_metrics.find("Recall") != -1 else 0.05   # TODO: 根据具体数值调整到最佳的显示效果
    y_tick = [i + y_tick_diff for i in np.arange(0, y_max+y_tick_diff, y_tick_diff)]

    plt.xticks(x_tick, x_tick_label, fontsize=15)
    plt.yticks(y_tick, fontsize=15)
    plt.ylabel("{}@{}".format(evaluation_metrics, cut_off), fontsize=20)

    plt.savefig("{}@{}.png".format(evaluation_metrics, cut_off))
    plt.show()


if __name__ == "__main__":
    method_to_recall20_cikm16 = {
        'Pop': [0.016537051935282022],
        'S-Pop': [0.2577098417806382],
        'Item-KNN': [0.42165012961473136],
        'BPR': [0.07169035487619559],
        'RN': [0.64241517071474102],
        'BRN': [0.65221313141414441],
    }
    present(method_to_recall20_cikm16, "CIKM-Recall", 20)
    # method_to_mrr20_cikm = {
    #     'Pop': [0.003338806995629901, 0.0034356056418846474, 0.004441449065699882],
    #     'S-Pop': [0.17905314890653487, 0.17555224549619722, 0.20112755289747455],
    #     'Item-KNN': [0.13258788973854033, 0.14001426756898144, 0.15058404162221356],
    #     'BPR': [0.02088768559774426, 0.022172372114107722, 0.026740737839360703]
    # }
    # present(method_to_mrr20_cikm, "CIKM-MRR", 20)
    #
    # method_to_recall20_rsc_64 = {
    #     'Pop': [0.061669435215946845, 0.06223639543256867, 0.057523494784674174],
    #     'S-Pop': [0.3033637873754153, 0.3059356033329904, 0.2890633068263968],
    #     'Item-KNN': [0.5279277408637874, 0.5370846620718033, 0.5422906124135082],
    #     'BPR': [0.09177740863787376, 0.08497068202859788, 0.08860890219973148]
    # }
    # present(method_to_recall20_rsc_64, "RSC64-Recall", 20)
    # method_to_mrr20_rsc_64 = {
    #     'Pop': [0.01291602772643783, 0.013048647308224033, 0.013068424013358318],
    #     'S-Pop': [0.20160906519405425, 0.19598335329758632, 0.19037347418823067],
    #     'Item-KNN': [0.23345500655806894, 0.23295481810209745, 0.23354105373434533],
    #     'BPR': [0.021262543151581426, 0.020236547575961928, 0.02078353517167078]
    # }
    # present(method_to_mrr20_rsc_64, "RSC64-MRR", 20)
    #
    # # 10
    # method_to_recall10_cikm16 = {
    #     'Pop': [0.008880243572395129, 0.00862658011615989, 0.010637346920532762],
    #     'S-Pop': [0.23832882273342354, 0.2307823710283567, 0.2535979261642978],
    #     'Item-KNN': [0.2847598105548038, 0.30005124701059105, 0.31259497631179045],
    #     'BPR': [0.039411366711772665, 0.03886231636487872, 0.048091534817198536]
    # }
    # present(method_to_recall10_cikm16, "CIKM-Recall", 10)
    # method_to_mrr10_cikm = {
    #     'Pop': [0.003045024110230471, 0.0030709228800277666, 0.004048527131123004],
    #     'S-Pop': [0.17935252754687775, 0.1738862587513079, 0.2001876114707971],
    #     'Item-KNN': [0.12428414363038848, 0.13187282742501355, 0.14239228214154456],
    #     'BPR': [0.01936950270635993, 0.020696572957847294, 0.025095987264128854]
    # }
    # present(method_to_mrr10_cikm, "CIKM-MRR", 10)
    #
    # method_to_recall10_rsc_64 = {
    #     'Pop': [0.040801495016611296, 0.0425882110893941, 0.0420324279665393],
    #     'S-Pop': [0.2871677740863787, 0.2900936117683366, 0.27615408447795103],
    #     'Item-KNN': [0.4361503322259136, 0.43359736652607755, 0.4389135598471548],
    #     'BPR': [0.05139119601328904, 0.04968624627095978, 0.05153361561499535]
    # }
    # present(method_to_recall10_rsc_64, "RSC64-Recall", 10)
    # method_to_mrr10_rsc_64 = {
    #     'Pop': [0.011510573221536686, 0.011706859474578852, 0.011994675662960301],
    #     'S-Pop': [0.1983298479275429, 0.19418437093316207, 0.18857029747766071],
    #     'Item-KNN': [0.22647190416864396, 0.22491517300950492, 0.22576607013764904],
    #     'BPR': [0.01836499861572534, 0.017712389965758933, 0.01815466969603082]
    # }
    # present(method_to_mrr10_rsc_64, "RSC64-MRR", 10)
