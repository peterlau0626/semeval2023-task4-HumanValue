#!/bin/bash
# **********************************************************
#
# * Author        : liupingan
# * Email         : lpanibupt@gmail.com
# * Create time   : 2022-09-30 10:53
# * Filename      : predict_kfold.py
# * Description   : 
#
# **********************************************************
import os
import sys

import traceback
import pandas as pd
import json
import csv
import collections

from model import predict_svm
from model import predict_bert_model

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
    ma_f1 = 0.0
    if ma_prec + ma_rec != 0:
        ma_f1 = 2 * (ma_prec * ma_rec) / (ma_prec + ma_rec) ### 求所有列的平均F1值
    sum_acc = sum_acc / float(col_num - 1) ### 求所有列的平均准确率

    print("precise:%f, recall:%f, f1:%f, accuracy:%f" % (ma_prec, ma_rec, ma_f1, sum_acc))
    #print("accuracy:%f, precise:%f, recall:%f, f1:%f" % (sum_acc, ma_prec, ma_rec, ma_f1))
    return ma_f1


def load_arguments_from_tsv(filepath, default_usage='test'):
    """
        Reads arguments from tsv file

        Parameters
        ----------
        filepath : str
            The path to the tsv file
        default_usage : str, optional
            The default value if the column "Usage" is missing

        Returns
        -------
        pd.DataFrame
            the DataFrame with all arguments

        Raises
        ------
        MissingColumnError
            if the required columns "Argument ID" or "Premise" are missing in the read data
        IOError
            if the file can't be read
        """
    try:
        dataframe = pd.read_csv(filepath, encoding='utf-8', sep='\t', header=0)
        if not {'Argument ID', 'Premise'}.issubset(set(dataframe.columns.values)):
            raise MissingColumnError('The argument "%s" file does not contain the minimum required columns [Argument ID, Premise].' % filepath)
        if 'Usage' not in dataframe.columns.values:
            dataframe['Usage'] = [default_usage] * len(dataframe)
        return dataframe
    except IOError:
        traceback.print_exc()
        raise

def load_json_file(filepath):
    """Load content of json-file from `filepath`"""
    with open(filepath, 'r') as  json_file:
        return json.load(json_file)

def load_values_from_json(filepath):
    """Load values per level from json-file from `filepath`"""
    json_values = load_json_file(filepath)
    #print(json_values)
    category_map = dict()
    level1_list = list()
    level2_list = list()
    for key in json_values:
        level2_list.append(key)
        values = json_values[key]
        for value in values:
            level1_list.append(value)
    category_map[1] = level1_list
    category_map[2] = level2_list    
    print("category1:%d, category2:%d" % (len(level1_list), len(level2_list)))
    return category_map

def load_labels_from_tsv(filepath, label_order):
    """
        Reads label annotations from tsv file

        Parameters
        ----------
        filepath : str
            The path to the tsv file
        label_order : list[str]
            The listing and order of the labels to use from the read data

        Returns
        -------
        pd.DataFrame
            the DataFrame with the annotations

        Raises
        ------
        MissingColumnError
            if the required columns "Argument ID" or names from `label_order` are missing in the read data
        IOError
            if the file can't be read
        """
    try:
        dataframe = pd.read_csv(filepath, encoding='utf-8', sep='\t', header=0)
        dataframe = dataframe[['Argument ID'] + label_order]
        return dataframe
    except IOError:
        traceback.print_exc()
        raise
    except KeyError:
        raise MissingColumnError('The file "%s" does not contain the required columns for its level.' % filepath)


def combine_columns(df_arguments, df_labels):
    """Combines the two `DataFrames` on column `Argument ID`"""
    return pd.merge(df_arguments, df_labels, on='Argument ID')

def split_arguments(df_arguments):
    """Splits `DataFrame` by column `Usage` into `train`-, `validation`-, and `test`-arguments"""
    train_arguments = df_arguments.loc[df_arguments['Usage'] == 'train'].drop(['Usage'], axis=1).reset_index(drop=True)
    valid_arguments = df_arguments.loc[df_arguments['Usage'] == 'validation'].drop(['Usage'], axis=1).reset_index(drop=True)
    test_arguments = df_arguments.loc[df_arguments['Usage'] == 'test'].drop(['Usage'], axis=1).reset_index(drop=True)
    
    return train_arguments, valid_arguments, test_arguments

def create_dataframe_head(argument_ids, model_name):
    """
        Creates `DataFrame` usable to append predictions to it

        Parameters
        ----------
        argument_ids : list[str]
            First column of the resulting DataFrame
        model_name : str
            Second column of DataFrame will contain the given model name

        Returns
        -------
        pd.DataFrame
            prepared DataFrame
    """
    df_model_head = pd.DataFrame(argument_ids, columns=['Argument ID'])
    #df_model_head['Method'] = [model_name] * len(argument_ids)

    return df_model_head

def predict_svm_model(df_test, model_dir, values, output_dir):
    num_levels = len(values)
    for i in range(num_levels):
        df_svm = create_dataframe_head(df_test['Argument ID'], model_name='SVM')
        level = i + 1
        print("===> SVM: Predicting Level %s..." % level)
        result = predict_svm(df_test, values[level],
                             os.path.join(model_dir, 'svm/svm_train_level{}_vectorizer.json'.format(level)),
                             os.path.join(model_dir, 'svm/svm_train_level{}_models.json'.format(level)))
        df_svm = pd.concat([df_svm, result], axis=1)
        
        print("level:%d, df_svm:" % level)
        print(df_svm.head(10))
        write_tsv_dataframe(os.path.join(output_dir, 'svm_predictions_level' + str(level) + ".tsv"), df_svm)        

def predict_bert(df_test, model_dir, values, output_dir, batch_size=12, add_conlusion_flag=False):
    #batch_size = 12
    level = 2
    df_bert = create_dataframe_head(df_test['Argument ID'], model_name='Bert')
    result = predict_bert_model(df_test, model_dir,
                    values[level], batch_size, add_label_flag = add_conclusion_flag)

    # write predict result
    columns = values[level]
    df_result = pd.DataFrame(result, columns=values[level])
    df_result.reset_index(inplace=True, drop=True)
    df_bert.reset_index(inplace=True, drop=True)

    df_bert = pd.concat([df_bert, df_result], axis=1)
    write_tsv_dataframe(output_dir, df_bert)

    return df_bert

def write_tsv_dataframe(filepath, dataframe):
    """
        Stores `DataFrame` as tsv file

        Parameters 
        ----------
        filepath : str
            Path to tsv file
        dataframe : pd.DataFrame
            DataFrame to store
              
        Raises
        ------
        IOError
            if the file can't be opened
    """       
    try:
        dataframe.to_csv(filepath, encoding='utf-8', sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE)
    except IOError:
        traceback.print_exc()   


if __name__ == '__main__':
    model_dir = "./model_files/Mao-zedong"
    data_dir = "./data/"
    output_dir = "./result/"
    value_json_filepath = data_dir + "value-categories.json"
    valid_set_dir1 = data_dir + "arguments-validation.tsv"
    valid_set_dir2 = data_dir + "arguments-validation-zhihu.tsv"

    #1. load json(it contains all level categories)
    values = load_values_from_json(value_json_filepath) 
    num_levels = len(values)
    print("========================================\nlevels")
    print(values)

    #2. load valid dataset
    df_valid1 = load_arguments_from_tsv(valid_set_dir1, default_usage='valid')
    df_valid2 = load_arguments_from_tsv(valid_set_dir2, default_usage='valid')
    df_valid = pd.concat([df_valid1, df_valid2])
    print("=======================================\nvalid_set")
    print(df_valid.head(10))
    print(df_valid.shape)


    #3. k-fold
    add_conclusion_flag = False
    batch_size = 24
    fold_num = 6
    f1_list = list()
    result_list = list()
    for k in range(1, fold_num+1):
        print("================================\nfold k:%d" % k)
        model_dir_fold = model_dir + "fold" + str(k)
        output_dir_fold = output_dir + "bert_predict_result_fold" + str(k) + ".csv"
        df_result = predict_bert(df_valid, model_dir_fold, values, output_dir_fold, batch_size, add_conclusion_flag)
        print(df_result.columns)
        print(df_result.head(10))
        result_list.append(df_result)

    print("length:%d" % len(result_list))
    
    #4. average k-fold predict result
    df_predict = pd.concat(result_list).groupby(by='Argument ID').mean().reset_index()
    origin_columns = df_predict.columns
    # 保留原始的样本顺序
    df_predict = df_valid.merge(df_predict, on='Argument ID')
    df_predict = df_predict[origin_columns]
    print("====================================\n average result")
    print(df_predict.head(10))
    print(df_predict.columns)

    columns = df_predict.columns
    for index, row in df_predict.iterrows():
        for col in df_predict.columns:
            if col == 'Argument ID':
                continue
            score = float(row[col])
            if score >= 0.5:
                df_predict.loc[index, (col)] = 1
            else:
                df_predict.loc[index, (col)] = 0

    print("====================================\n average result")
    print(df_predict.head(10))
    print(df_predict.columns)

    #5. save predict result
    df_predict[columns].to_csv(output_dir+"bert_predictions_level2.tsv", encoding='utf-8', sep='\t', index=False, header=columns, quoting=csv.QUOTE_NONE)
