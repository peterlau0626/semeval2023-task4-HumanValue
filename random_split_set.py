#!/bin/bash
# **********************************************************
#
# * Author        : liupingan
# * Email         : liupingan-jk@360shuke.com
# * Create time   : 2022-09-29 15:57
# * Filename      : random_split_set.py
# * Description   : 
#
# **********************************************************
import os
import sys
import random

argument_filepath="./data/arguments-training.tsv"
reader = open(argument_filepath, 'r')
lines = reader.readlines()
writer = open("./abc", 'w')
for i in range(0, len(lines)):
    line = lines[i]
    if i == 0:
        line = 'Argument ID\tPart\tUsage\tConclusion\tStance\tPremise'
        writer.write(line + "\n")
    else:
        items = line.strip().split('\t')
        if len(items) != 4:
            print("length not equal 4")
            continue
        aid = items[0].strip()
        con = items[1].strip()
        sta = items[2].strip()
        pre = items[3].strip()
        part = ''
        if aid[0] == 'A':
            part = 'usa'
        elif aid[0] == 'B':
            part = 'africa'
        elif aid[0] == 'C':
            part = 'china'
        elif aid[0] == 'D':
            part = 'india'
        else:
            print("part is error:%d" % aid)
        
        usage= ''
        rand = random.random()
        if rand >= 0 and rand < 0.8:
             usage = 'train'
        elif rand >= 0.8 and rand <= 0.95:
            usage = 'test'
        else:
            usage = 'validation'
        
        writer.write(aid + "\t" + part + "\t" +  usage + "\t" + con + "\t" + sta + "\t" + pre + "\n")
