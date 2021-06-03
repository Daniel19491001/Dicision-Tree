mport math
from collections import Counter
import numpy as np


# 计算熵
def calculate_entropy(step):
    res = -sum([(step[i] / sum(step)) * math.log((step[i] / sum(step)), 2) if step[i] != 0 else 0 for i in range(0, 3)])
    return res


#  计算连续属性的信息熵
def getentropy(sort_list, sort_label, cutoff1, cutoff2):
    assert cutoff1 not in sort_list
    assert cutoff2 not in sort_list
    assert cutoff2 > cutoff1
    assert sort_list[0] <= sort_list[-1]
    step1 = [0,0,0]  # < cutoff1
    step2 = [0,0,0]  #  1-2
    step3 = [0,0,0]  #  2-
    for i in range(len(sort_list)):
        if sort_list[i] < cutoff1:
            if sort_label[i] == 'setosa':
                step1[0] += 1
            elif sort_label[i] == 'versicolor':
                step1[1] += 1
            elif sort_label[i] == 'virginica':
                step1[2] += 1
        elif sort_list[i] <cutoff2 and sort_list[i] > cutoff1:
            if sort_label[i] == 'setosa':
                step2[0] += 1
            elif sort_label[i] == 'versicolor':
                step2[1] += 1
            elif sort_label[i] == 'virginica':
                step2[2] += 1
        else:
            if sort_label[i] == 'setosa':
                step3[0] += 1
            elif sort_label[i] == 'versicolor':
                step3[1] += 1
            elif sort_label[i] == 'virginica':
                step3[2] += 1
    w1 = sum(step1)/150
    w2 = sum(step2)/150
    w3 = sum(step3)/150
    entropy = w1*calculate_entropy(step1) + w2*calculate_entropy(step2) + w3*calculate_entropy(step3)
    if round(cutoff1,3)==0.95 and round(cutoff2,3)==1.786:
        print("属性4的最小分割:", step1, step2, step3)
    return entropy


# 选择分割点
def select_cutoff(feature_list, feature_label):
    # print(feature_list)
    sort_index = np.argsort(feature_list)
    # print(sort_index)
    # 排序
    sort_feature = [0 for i in feature_list]
    sort_label = [0 for i in feature_label]
    for i in range(len(sort_index)):
        sort_feature[i] = feature_list[sort_index[i]]
        sort_label[i] = feature_label[sort_index[i]]
    # print(sort_feature,"\n", sort_label)
    # 获取分界点1  2
    ct = Counter(sort_feature)
    removesame = sorted(ct.most_common())  # 从小到大排序
    cut_off = []
    for i in range(len(removesame)-1):
        w1 = removesame[i][1]/(removesame[i][1]+removesame[i+1][1])
        w2 = removesame[i+1][1]/(removesame[i][1]+removesame[i+1][1])
        cut_off.append(w1*removesame[i][0]+w2*removesame[i+1][0])
    cut_off_choose = []
    for i in range(len(cut_off)-1):
        for j in range(i+1, len(cut_off)):
            cut_off_choose.append((cut_off[i], cut_off[j]))  # 元组
    # 逐次计算熵
    res = []
    for cutoff in cut_off_choose:
        res.append(getentropy(sort_feature, sort_label, cutoff[0], cutoff[1]))
    return cut_off_choose, res


def get_min_entropy_cutoff(cut_off_choose, res):
    res_ = [round(eve,5) for eve in res]
    tp = min(res_)
    i = res_.index(tp)
    res_list = [round(eve,3) for eve in list(cut_off_choose[i])]
    return res_list

# 获取最优分割点
def main_of_get_cut_off():
    # read iris data-set as txt.
    with open(r"iris.txt","r",encoding="utf-8") as file:
        all_data = [line.split() for line in file]
    attribute_name = all_data[0]
    data = [[float(number) for number in sample[1:-1]] for sample in all_data[1:]]
    labels = [sample[-1][1:-1] for sample in all_data[1:]]
    sepal_length = [eve[0] for eve in data]
    sepal_width= [eve[1] for eve in data]
    petal_length= [eve[2] for eve in data]
    petal_width= [eve[3] for eve in data]
    # 依次选择四个特征的值进行处理，选择每个特征的分割点
    cut_off_choose1, res1 = select_cutoff(sepal_length, labels)
    cut_off_choose2, res2 = select_cutoff(sepal_width, labels)
    cut_off_choose3, res3 = select_cutoff(petal_length, labels)
    cut_off_choose4, res4 = select_cutoff(petal_width, labels)
    print("四个属性的最小熵：", min(res1), min(res2), min(res3), min(res4))
    res = [get_min_entropy_cutoff(cut_off_choose1, res1),get_min_entropy_cutoff(cut_off_choose2, res2),
            get_min_entropy_cutoff(cut_off_choose3, res3),get_min_entropy_cutoff(cut_off_choose4, res4)]
    return res, data, labels


if __name__ == '__main__':

    razors, data, labels_ = main_of_get_cut_off()
    labels = []
    print("四个属性的分割点（按顺序）：", razors)
