import torch
from transformers import (AutoTokenizer, AutoModel)
from transformers.optimization import get_cosine_schedule_with_warmup
import pandas as pd
from datasets import (Dataset, DatasetDict, load_dataset)
import numpy as np
import torch.nn as nn
import os
import math
from sklearn.metrics import f1_score
from torch.utils.data import  DataLoader
from tqdm import tqdm
import random
import torch.distributed as dist
import torch.multiprocessing as mp


#tokenizer=AutoTokenizer.from_pretrained('roberta-large')
#device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#label_describe=['Self-direction: thought Be creative Be curious Have freedom of thought', 'Self-direction: action Be choosing own goals Be independent Have freedom of action Have privacy', 'Stimulation Have an exciting life Have a varied life Be daring', 'Hedonism Have pleasure', 'Achievement Be ambitious Have success Be capable Be intellectual Be courageous', 'Power: dominance Have influence Have the right to command', 'Power: resources Have wealth', 'Face Have social recognition Have a good reputation', 'Security: personal Have a sense of belonging Have good health Have no debts Be neat and tidy Have a comfortable life', 'Security: societal Have a safe country Have a stable society', 'Tradition Be respecting traditions Be holding religious faith', 'Conformity: rules Be compliant Be self-disciplined Be behaving properly', 'Conformity: interpersonal Be polite Be honoring elders', 'Humility Be humble Have life accepted as is', 'Benevolence: caring Be helpful Be honest Be forgiving Have the own family secured Be loving', 'Benevolence: dependability Be responsible Have loyalty towards friends', 'Universalism: concern Have equality Be just Have a world at peace', 'Universalism: nature Be protecting the environment Have harmony with nature Have a world of beauty', 'Universalism: tolerance Be broadminded Have the wisdom to accept others', 'Universalism: objectivity Be logical Have an objective view']
#label_info=tokenizer(label_describe, truncation=True,padding=True)
#print(label_info)
#
#for key in label_info.keys():
#    label_info[key]=torch.tensor(label_info[key],dtype=torch.int32)
#print(label_info)
#
#
#base_model=AutoModel.from_pretrained('roberta-large')
#
#
#labels_feature = base_model(**label_info).pooler_output
#print('----------------------')
#print(labels_feature)
#print(labels_feature.shape)
#
#embed=torch.nn.Embedding(20, base_model.config.hidden_size)
#embed.weight = torch.nn.Parameter(labels_feature)
#print('======================')
#print(embed)
#print(embed.weight)
#
