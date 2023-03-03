#!/bin/bash
# **********************************************************
#
# * Author        : liupingan
# * Email         : lpanibupt@gmail.com
# * Create time   : 2022-09-26 13:15
# * Filename      : evaluate.py
# * Description   : 
#
# **********************************************************
import os
import sys
import pandas as pd
import collections


def metric(label_set, predict_set):
    value_names = list(label_set.columns)
    row_num = predict_set.shape[0]
    col_num = predict_set.shape[1]
    if row_num != label_set.shape[0] or col_num != label_set.shape[1]:
        print("shape not equal")
        sys.exit(-1)

    ma_top_list = collections.OrderedDict()
    ma_rec_bottom_list = collections.OrderedDict()
    ma_prec_bottom_list = collections.OrderedDict()
    correct_Nums_list = collections.OrderedDict()

    for j in range(0, col_num):
        ma_top_list[j] = 0
        ma_rec_bottom_list[j] = 0
        ma_prec_bottom_list[j] = 0
        correct_Nums_list[j] = 0


    print("shape:%d,%d" % (row_num, col_num))
    for i in range(0, row_num):
        for j in range(1, col_num):
            if label_set.iat[i, j] == 1:
                #print("i:%d,j:%d" % (i, j))
                ma_rec_bottom_list[j] += 1 ### 每一列，有多少真实为1的label.
                if predict_set.iat[i, j] == 1:
                    ma_top_list[j] += 1 ### 每一列，在真实为1的label中，我预测为1的有多少
                    ma_prec_bottom_list[j] += 1  ### 不管真实label是不是1，我预测为1的有多少
                    correct_Nums_list[j] += 1 ### 预测对的数量，包含label=1和label=0都预测对的
            else:
                if predict_set.iat[i, j] == 1:
                    ma_prec_bottom_list[j] += 1
                else:
                    correct_Nums_list[j] += 1


    ma_rec_list = collections.OrderedDict()
    ma_prec_list = collections.OrderedDict()
    ma_f1_list = collections.OrderedDict()
    accuracy_list = collections.OrderedDict()

    ma_rec = 0
    ma_prec = 0
    sum_acc = 0

    for j in range(1, col_num):
        if ma_rec_bottom_list[j] == 0:
            ma_rec_list[j] = 0.0
        else:
            ma_rec_list[j] = ma_top_list[j] / float(ma_rec_bottom_list[j]) ### 所有真实为1的样本中，有多少是我认为为1的。为recall召回率 
        ma_rec += ma_rec_list[j]

        if ma_prec_bottom_list[j] == 0:
            ma_prec_list[j] = 0.0
        else:
            ma_prec_list[j] = ma_top_list[j] / float(ma_prec_bottom_list[j]) ### 所有我预测为1的样本里面，有多少真的是1的。为precise精确率 
        ma_prec += ma_prec_list[j]

        Sum = ma_prec_list[j] + ma_rec_list[j]
        if Sum == 0:
            ma_f1_list[j] = 0.0
        else:
            ma_f1_list[j] = 2 * ma_prec_list[j] * ma_rec_list[j] / float(Sum) ### 为F1值
        
        accuracy_list[j] = correct_Nums_list[j] / float(row_num) ### 对于每一列, 预测对的样本数/所有样本数。为accuracy准确率
        sum_acc += accuracy_list[j]

        print(value_names[j] + "," + str(round(accuracy_list[j], 4)) + "," + str(round(ma_prec_list[j], 4)) + "," + str(round(ma_rec_list[j], 4)) + "," + str(round(ma_f1_list[j], 4)))

    ma_rec = ma_rec / float(col_num - 1) ### 求所有列的平均召回率
    ma_prec = ma_prec / float(col_num - 1)  ### 求所有列的平均精确率
    ma_f1 = 2 * (ma_prec * ma_rec) / (ma_prec + ma_rec) ### 求所有列的平均F1值
    sum_acc = sum_acc / float(col_num - 1) ### 求所有列的平均准确率

    print("precise:%f, recall:%f, f1:%f, accuracy:%f" % (ma_prec, ma_rec, ma_f1, sum_acc))
    #print("accuracy:%f, precise:%f, recall:%f, f1:%f" % (sum_acc, ma_prec, ma_rec, ma_f1))



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("eg. python Evaluate.py ./data/labels-merge-validation.tsv ./result/bert_predictions_level2.tsv")
        print("eg. python Evaluate.py ./data/bak/test_labels_level2.tsv ./result/bert_predictions_level2.tsv")
        sys.exit(-1)

    label_file = sys.argv[1]
    label_set = pd.read_csv(label_file, header=0, sep="\t")
    print(label_set.columns)
    a = list(label_set.columns)
    print(a)
    print(len(a))

    predict_file = sys.argv[2]
    predict_set = pd.read_csv(predict_file, header=0, sep="\t")
    
    metric(label_set, predict_set)
