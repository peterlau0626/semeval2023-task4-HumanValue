#!/bin/bash
# **********************************************************
#
# * Author        : liupingan
# * Email         : liupingan-jk@360shuke.com
# * Create time   : 2022-09-29 14:23
# * Filename      : preprocess.py
# * Description   : 
#
# **********************************************************
import os
import sys

import traceback
import pandas as pd
import json

from model import train_svm
from model import train_bert_model
from sklearn.model_selection import KFold

import random

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

def create_augmentor():
    #aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute")
    #aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action="insert")
    aug = naw.RandomWordAug(action="swap")
    #aug = naw.RandomWordAug()
    #aug = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en')
    return aug


def get_augmented_text(text, aug):
    augmented_text = aug.augment(text)
    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text)
    return augmented_text

def augment(df_train_tmp, augment_rate=0.2):
    # if augment_rate=3.6, then multi = 3, rest = 0.6
    multi = int(augment_rate)
    rest = augment_rate - multi
    # create augmentor 
    aug = create_augmentor()

    df_aug_list = list()
    df_aug_list.append(df_train_tmp)
    
    for k in range(0, multi + 1):
        # copy original train_set
        df_aug = df_train_tmp.copy(deep=True)
        # if not the last loop, each sample generates a augmented sample. 
        if k != multi:
            for i in range(0, len(df_aug)):
                text = df_aug.iloc[i]['Premise']
                augmented_text = get_augmented_text(text, aug)
                df_aug.iloc[i]['Premise'] = augmented_text[0]
                df_aug.iloc[i]['Argument ID'] = df_aug.iloc[i]['Argument ID'] + "_aug" + str(k)
            df_aug_list.append(df_aug)
        # if the last loop, generate an augmented sample with prob rest(=0.6) 
        else:
            if rest == 0:
                continue
            for i in range(0, len(df_aug)):
                rand = random.random()
                if rand > rest:
                    #duplicate sample
                    df_aug.iloc[i]['Argument ID'] = "need remove"
                    continue
                text = df_aug.iloc[i]['Premise']
                augmented_text = get_augmented_text(text, aug)
                df_aug.loc[i, 'Premise'] = augmented_text[0]
                df_aug.iloc[i]['Premise'] = augmented_text[0]
                df_aug.iloc[i]['Argument ID'] = df_aug.iloc[i]['Argument ID'] + "_aug" + str(k)
            df_aug = df_aug.drop(df_aug[df_aug['Argument ID'] == "need remove"].index).reset_index(drop=True)
            df_aug_list.append(df_aug)

    df_train_all = pd.concat(df_aug_list)
    print("=================================\nsample augment:")
    print(df_train_tmp.head(10))
    print(df_train_tmp.shape)
    print(df_train_all.head(10))
    print(df_train_all.shape)
    return df_train_all

def loadPosWeight(weight_path):
    weight_list = list()
    reader = open(weight_path, 'r')
    lines = reader.readlines()
    for line in lines:
        items = line.strip().split(',')
        if len(items) != 2:
            continue
        weight_list.append(float(items[1].strip()))
    return weight_list

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


def train_svm_model(df_train_all, df_valid_all, values, model_dir):
    num_levels = len(values)
    for i in range(num_levels):
        level = i + 1
        df_train = df_train_all[i]
        value = values[level]
        svm_f1_scores = train_svm(df_train_all[i], values[level],
                        os.path.join(model_dir, 'svm/svm_train_level{}_vectorizer.json'.format(level)),
                        os.path.join(model_dir, 'svm/svm_train_level{}_models.json'.format(level)),
                        test_dataframe=df_valid_all[i])
        print("------------------------------------\n\n")
        print("F1-Scores for Level %s:" % level)
        #print(svm_f1_scores)
        sum_score = 0.0
        for category in svm_f1_scores:
            sum_score += svm_f1_scores[category]
            print("%s,%f" % (category, svm_f1_scores[category]))
        print("Average F1-score,%f" % (sum_score/len(svm_f1_scores)))

def train_bert(df_train_all,
        df_valid_all,
        values,
        pos_weight_list,
        model_dir,
        batch_size=12,
        epoch=20,
        add_conclusion_flag=False):

    print("===> Bert: Training Level 2...")
    level=2

    train_bert_model(df_train_all,
            model_dir,
            values[level],
            batch_size,
            pos_weight_list,
            test_dataframe=df_valid_all,
            num_train_epochs=epoch,
            add_label_flag=add_conclusion_flag)
    print("finish")



#    bert_model_evaluation = train_bert_model(df_train_all,
#                                    model_dir,
#                                    values[level],
#                                    batch_size,
#                                    pos_weight_list,
#                                    test_dataframe=df_valid_all,
#                                    num_train_epochs=epoch,
#                                    add_label_flag=add_conclusion_flag)
#    print("F1-Scores for Level %s:" % level)
#    print(bert_model_evaluation['eval_f1-score'])
#    print(bert_model_evaluation)



if __name__ == '__main__':
    levels = ["1", "2"]
    data_dir = "./data/"
    model_dir = "./model_files/"
    train_set_dir = data_dir + "arguments-training.tsv" 
    label_of_trainset_dir = data_dir + "labels-training.tsv"
    value_json_filepath = data_dir + "value-categories.json"     
    level = 2
    fold_k = 6
    augment_flag = False
    augment_rate = 1.0

    #1. load arguments
    df_train = load_arguments_from_tsv(train_set_dir, default_usage='train')
    print("=======================================\ntrain_set")
    print(df_train.head(10))
    print(df_train.shape)

    #2. load json(it contains all level categories)
    values = load_values_from_json(value_json_filepath) 
    num_levels = len(values)
    print("========================================\nlevels")
    print(values)
   

    #3. load pos weights
    pos_weight_list = list()
    weight_path = data_dir + "pos_weight2.dat"
    pos_weight_list = loadPosWeight(weight_path)
   
    print("load pos weight list:%d" % len(pos_weight_list))

    #4. load labels
    df_label_train = load_labels_from_tsv(label_of_trainset_dir, values[level]) 

    #5. combine arguments and labels
    df_all = combine_columns(df_train, df_label_train)
    print("=======================================\ncombine train arguments and labels")
    print(df_all.head(10))
    print(df_all.shape)

    #6. k-fold
    kf = KFold(n_splits = fold_k, shuffle = True, random_state = 2)
    i = 0
    for train_index , valid_index in kf.split(df_all):
        i += 1
        model_dir_fold = model_dir + "fold" + str(i)
        if not os.path.exists(model_dir_fold):
            os.mkdir(model_dir_fold)
        print("==================================\nfold k=%d" % i)
        df_train_all = df_all.iloc[train_index, :]
        if augment_flag:
            print("start augment sample")
            df_train_all = augment(df_train_all, augment_rate)
        df_valid_all = df_all.iloc[valid_index, :]

        # train bert
        batch_size = 24
        epoch = 30
        add_conclusion_flag = False
        train_bert(df_train_all,
                df_valid_all,
                values,
                pos_weight_list,
                model_dir_fold,
                batch_size,
                epoch,
                add_conclusion_flag)
        
        # save valid set for evaluation
        df_valid_all.to_csv(model_dir_fold + "/valid.csv", columns=df_valid_all.columns, index=False, sep="\t")
        df_train_all.to_csv(model_dir_fold + "/train.csv", columns=df_train_all.columns, index=False, sep="\t")

    print("======================================\ndone")
