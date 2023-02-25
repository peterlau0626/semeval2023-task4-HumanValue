# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 11:50:15 2022

@author: 12695
"""
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

#model_dir="../"
#base_model=AutoModel.from_pretrained('roberta-large')
tokenizer=AutoTokenizer.from_pretrained('roberta-large')
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_describe=['Self-direction: thought Be creative Be curious Have freedom of thought', 'Self-direction: action Be choosing own goals Be independent Have freedom of action Have privacy', 'Stimulation Have an exciting life Have a varied life Be daring', 'Hedonism Have pleasure', 'Achievement Be ambitious Have success Be capable Be intellectual Be courageous', 'Power: dominance Have influence Have the right to command', 'Power: resources Have wealth', 'Face Have social recognition Have a good reputation', 'Security: personal Have a sense of belonging Have good health Have no debts Be neat and tidy Have a comfortable life', 'Security: societal Have a safe country Have a stable society', 'Tradition Be respecting traditions Be holding religious faith', 'Conformity: rules Be compliant Be self-disciplined Be behaving properly', 'Conformity: interpersonal Be polite Be honoring elders', 'Humility Be humble Have life accepted as is', 'Benevolence: caring Be helpful Be honest Be forgiving Have the own family secured Be loving', 'Benevolence: dependability Be responsible Have loyalty towards friends', 'Universalism: concern Have equality Be just Have a world at peace', 'Universalism: nature Be protecting the environment Have harmony with nature Have a world of beauty', 'Universalism: tolerance Be broadminded Have the wisdom to accept others', 'Universalism: objectivity Be logical Have an objective view']
class Attention(nn.Module):
    def __init__(self,d_model, d_k, d_v, num_labels):
        super(Attention, self).__init__()
        self.d_k=d_k
        self.d_v=d_v
#        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        self.fc1 = nn.Linear(d_v, d_model, bias=False)
        self.LayerNorm1=nn.LayerNorm(d_model)
        self.num_labels=num_labels
        
        self.output_layer = nn.Linear(d_model, num_labels)
        self.embedding_dropout = nn.Dropout(p=0.3)

    def forward(self, n_heads, input_Q, input_K, input_V, attn_mask):
        input_K= self.embedding_dropout(input_K)
        input_V= self.embedding_dropout(input_K)
        residual, batch_size = input_Q, input_K.size(0)
        Q = input_Q.view(-1, n_heads, int(self.d_k/n_heads)).transpose(0, 1)    # Q: [n_heads, len_label, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, int(self.d_k/n_heads)).transpose(1, 2)    # K: [batch_size, n_heads, len_sentence, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, int(self.d_v/n_heads)).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1 , 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(int(self.d_k/n_heads))    # scores : [batch_size, n_heads, len_label, len_sentence]
        scores.masked_fill_(attn_mask.eq(0), -1e9)                            # 如果时停用词P就等于 0
        attn = nn.Softmax(dim=-1)(scores)
#        attn= scores.sigmoid()
        context = torch.matmul(attn, V)                                 # [batch_size, n_heads, len_label, d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.d_v)                    # context: [batch_size, len_label, n_heads * d_v]
        context = self.fc1(context)
        context = self.LayerNorm1(context)  # [batch_size, len_label, d_model]
        avg_sentence_embeddings = torch.sum(context, 1)/self.num_labels
        pred = self.output_layer(avg_sentence_embeddings)
#        residual=output
#        output = self.fc2(output)
#        output = self.LayerNorm2(output+residual)
        return pred,avg_sentence_embeddings

class MyModel(nn.Module):
    def __init__(self, base_model, num_labels,label_info,freeze_bert=False):
        super(MyModel, self).__init__()
        self.base_model=base_model
        self.num_labels=num_labels
        if freeze_bert:                 #冻结bert的参数不更新
            for p in self.base_model.parameters():
                p.requires_grad = False
        
        self.attention=Attention(base_model.config.hidden_size , base_model.config.hidden_size, base_model.config.hidden_size, num_labels)
        
        self.label_embed = self.load_labelembedd(label_info)
        
        
    def load_labelembedd(self, label_info):
        labels_feature=self.base_model(**label_info).pooler_output
        embed=torch.nn.Embedding(self.num_labels,self.base_model.config.hidden_size)
        embed.weight = torch.nn.Parameter(labels_feature)
        return embed
        
    def forward(self,inputs):
        setence_feature=self.base_model(**inputs).last_hidden_state 
        #bert的输出包括(last_hidden_state,pooler_output,hidden_states,attentions)
        #labels_feature=self.base_model(**label_info).pooler_output
        labels_feature=self.label_embed.weight.data
        attn_mask=inputs['attention_mask'] 
        attn_mask=attn_mask.unsqueeze(1)
        attn_mask=attn_mask.expand(setence_feature.size(0), labels_feature.size(0), setence_feature.size(1))
        
        pred,embed=self.attention(16,labels_feature,setence_feature,setence_feature,attn_mask)
        
        return pred,embed
    
class CLloss(nn.Module):
    def __init__(self):
        super(CLloss, self).__init__()
        self.t=0.1
    def forward(self,embed,labels):    
        labels=labels.float()
        embed=nn.functional.normalize(embed, dim=-1)
        batch_size=embed.size(0)
        t=self.t
        loss=torch.tensor(0.0).to(device)
        for i in range(batch_size):
            Ci=1e-12
            Ei=1e-12
            Li=0.0
            for j in range(batch_size):
                if i==j:
                    continue
                else:
                    Ci+=torch.matmul(labels[i],labels[j])
                    Ei+=torch.exp(-nn.PairwiseDistance(p=2)(embed[i],embed[j])/t)
                    
            for j in range(batch_size):
                if i==j:
                    continue
                else:
                    Cij=torch.matmul(labels[i],labels[j])/Ci
                    Eij=-Cij*torch.log(torch.exp(-nn.PairwiseDistance(p=2)(embed[i],embed[j])/t)/Ei)
                    Li+=Eij
                    
            loss+=Li/batch_size
            
            return loss
    
    
def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=False):
    """Compute accuracy of predictions"""
    #y_pred = torch.from_numpy(y_pred)
    #y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()

    return ((y_pred > thresh) == y_true.bool()).float().mean().item()


def f1_score_per_label(y_pred, y_true, value_classes, thresh=0.5, sigmoid=False):
    """Compute label-wise and averaged F1-scores"""
    #y_pred = torch.from_numpy(y_pred)
    #y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()

    y_true = y_true.bool().numpy()
    y_pred = (y_pred > thresh).numpy()

    f1_scores = {}
    for i, v in enumerate(value_classes):
        f1_scores[v] = round(f1_score(y_true[:, i], y_pred[:, i], zero_division=0), 6)

    f1_scores['avg-f1-score'] = round(np.mean(list(f1_scores.values())), 6)

    return f1_scores


def compute_metrics(eval_pred, value_classes):
    """Custom metric calculation function for MultiLabelTrainer"""
    predictions, labels = eval_pred
    f1scores = f1_score_per_label(predictions, labels, value_classes)
    return {'accuracy_thresh': accuracy_thresh(predictions, labels), 'f1-score': f1scores,
            'marco-avg-f1score': f1scores['avg-f1-score']}

def my_collate(batch):
    '''
    数据集根据batch_size堆叠起来输入到模型的函数
    返回值是一个dict，键包括labels和tokenizer后的结果，值类型为tensor
    eg:{labels:tensor(),input_ids:tensor(),attention_mask:tensor()}
    '''
    keys=batch[0].keys()
    sentences=[x['input_ids'] for x in batch]
    batch_len=len(sentences)
#    max_length=max([len(s) for s in sentences])
    max_length=256
    batch_dict={}
    for key in keys:
        if key=='labels':
            batch_dict[key]=np.zeros((batch_len, len(batch[0][key])),dtype=np.int32)
        else:
            batch_dict[key]=np.zeros((batch_len, max_length),dtype=np.int32)
        for i in range(batch_len):
            batch_dict[key][i][:len(batch[i][key])]=batch[i][key]
        batch_dict[key]=torch.tensor(batch_dict[key],dtype=torch.int32)
    return batch_dict

def tokenize_and_encode(examples):
    """Tokenizes each arguments "Premise" """
    return tokenizer(examples['Premise'], truncation=True)    

def convert_to_dataset(train_dataframe, test_dataframe, labels):
    """
        这里需要注意输入的labels的顺序和返回的cols顺序，在训练时和测试时都应该保持一致
        Converts pandas DataFrames into a DatasetDict

        Parameters
        ----------
        train_dataframe : pd.DataFrame
            Arguments to be listed as "train"
        test_dataframe : pd.DataFrame
            Arguments to be listed as "test"
        labels : list[str]
            The labels in both DataFrames

        Returns
        -------
        tuple(DatasetDict, list[str])
            a `DatasetDict` with attributes "train" and "test" for the listed arguments,
            a `list` with the contained labels
        """
    # 把premise和所有label对应的columns名称取出来
    print("execute convert_to_dataset.")
    column_intersect = [x for x in (['Premise'] + labels) if x in train_dataframe.columns.values]
    
    # 生成dict, 'Premise': ['Premise1', 'Premise2', ...], 'label1':[1, 0, 1...], 'label2':[1,0,0,...], 'label3':[1,0,0,...] ...
    #print((train_dataframe[column_intersect]).to_dict('list'))
    train_dataset = Dataset.from_dict((train_dataframe[column_intersect]).to_dict('list'))
    test_dataset = Dataset.from_dict((test_dataframe[column_intersect]).to_dict('list'))


    ds = DatasetDict()
    ds['train'] = train_dataset
    ds['test'] = test_dataset


    ds = ds.map(lambda x: {"labels": [int(x[c]) for c in ds['train'].column_names if
                                      c not in ['Argument ID', 'Conclusion', 'label_list', 'Stance', 'Premise', 'Part']]})

    cols = ds['train'].column_names
    cols.remove('labels')
    
    #做了tokenize编码
    ds_enc = ds.map(tokenize_and_encode, batched=True, remove_columns=cols)
    cols.remove('Premise')
    
    return ds_enc,  cols

def get_KNNpred(embed,x,y):
    k=32
    
    total_num=x.size(0)
    batch_size=embed.size(0)
    
    y_pred=torch.tensor([]).to(device)
    values=torch.tensor([]).to(device)
    for i in range(total_num):
        value=torch.exp(-nn.PairwiseDistance(p=2)(embed,x[i]))
        value=value.unsqueeze(0)
        values=torch.cat((values,value) ,dim=0) #(total_num,batch_size)
    
    values=values.transpose(0, 1)        #(batch_size,total_num)
    for i  in range(batch_size):
        sum=0.0
        value,idx=torch.sort(values[i], descending=True)
        logist=torch.zeros(y.size(1)).to(device)
        for j in range(k):
            sum+=value[j]
        for j in range(k):
            logist+=value[j]/sum*y[idx[j].item()]
        logist=logist.unsqueeze(0)
        y_pred=torch.cat((y_pred,logist), dim=0)
    return y_pred

def run_eval(model,valid_loader,label_info,value_classes,x,y):
    model.eval()
    y_pred=torch.tensor([]).to(device)
    y_true=torch.tensor([]).to(device)
    
    with torch.no_grad():
            
        for idx,batch_samples in enumerate(tqdm(valid_loader)):
            
           for key in batch_samples.keys():
               batch_samples[key]=batch_samples[key].to(device)
               
           labels=batch_samples.pop('labels') 
           output,embed=model(batch_samples)
           KNNpred=get_KNNpred(embed,x,y)
           
           output=output.sigmoid()
           output=0.75*output+0.25*KNNpred
           y_pred=torch.cat((y_pred,output), dim=0)
           y_true=torch.cat((y_true,labels), dim=0)
           
        y_pred=y_pred.to(torch.device('cpu'))     #预测的结果放回CPU上方便后面计算
        y_true=y_true.to(torch.device('cpu'))
        
        eval_pred=(y_pred,y_true)
        #print(y_pred)
        metrics=compute_metrics(eval_pred, value_classes)
        return metrics
   
def add_conclusion(train_dataframe):
    for i in range(len(train_dataframe)):
        if train_dataframe.iloc[i]['Stance'] =='against':
            train_dataframe.loc[i ,'Premise']= 'I disagree that '+train_dataframe.iloc[i]['Conclusion']+', because '+train_dataframe.iloc[i]['Premise']
        else:
            train_dataframe.loc[i ,'Premise']= 'I agree that '+train_dataframe.iloc[i]['Conclusion']+', because '+train_dataframe.iloc[i]['Premise']
             
def train_bert_model(train_dataframe,
                    model_dir,
                    labels,
                    batch_size,
                    weight_list=[],
                    test_dataframe=None,
                    num_train_epochs=20,
                    add_label_flag = False):
    """
        Trains Bert model with the arguments in `train_dataframe`

        Parameters
        ----------
        train_dataframe: pd.DataFrame
            The arguments to be trained on
        model_dir: str
            The directory for storing the trained model
        labels : list[str]
            The labels in the training data
        test_dataframe: pd.DataFrame, optional
            The validation arguments (default is None)
        num_train_epochs: int, optional
            The number of training epochs (default is 20)

        Returns
        -------
        Metrics
            result of validation if `test_dataframe` is not None
        NoneType
            otherwise
        """
    if test_dataframe is None:
        test_dataframe = train_dataframe
        train_dataframe=train_dataframe.reset_index(drop=True)
        add_conclusion(train_dataframe)
    else:
        train_dataframe=train_dataframe.reset_index(drop=True)
        test_dataframe=test_dataframe.reset_index(drop=True)
        add_conclusion(train_dataframe)
        add_conclusion(test_dataframe)
    
    lr=1e-5
    weight_decay=0.01
    
    ds, value_classes = convert_to_dataset(train_dataframe, test_dataframe, labels)
    #print(ds)
    
    
    label_info=tokenizer(label_describe, truncation=True,padding=True)
    
    for key in label_info.keys():
        label_info[key]=torch.tensor(label_info[key],dtype=torch.int32)
    
    train_loader=DataLoader(ds['train'],batch_size=batch_size,shuffle=True,collate_fn=my_collate)
    valid_loader=DataLoader(ds['test'],batch_size=batch_size,shuffle=False,collate_fn=my_collate)

    base_model=AutoModel.from_pretrained('roberta-large')
    model=MyModel(base_model, len(value_classes),label_info,freeze_bert=False)
    
    model=model.to(device)
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in list(model.attention.named_parameters()) if not any(nd in n for nd in no_decay)],
         'lr': lr*20, 'weight_decay':weight_decay*5},
        {'params': [p for n, p in list(model.attention.named_parameters()) if any(nd in n for nd in no_decay)],
         'lr': lr*20, 'weight_decay': 0.0},
        {'params': [p for n, p in list(model.label_embed.named_parameters()) if not any(nd in n for nd in no_decay)],
         'lr': lr*20, 'weight_decay':weight_decay*5},
        {'params': [p for n, p in list(model.label_embed.named_parameters()) if any(nd in n for nd in no_decay)],
         'lr': lr*20, 'weight_decay': 0.0}
        
    ]
    
    optimizer_grouped_parameters2 = [
        {'params': [p for n, p in list(model.base_model.named_parameters()) if not any(nd in n for nd in no_decay)],
         'lr': lr,'weight_decay': weight_decay},
        {'params': [p for n, p in list(model.base_model.named_parameters()) if any(nd in n for nd in no_decay)],
         'lr': lr,'weight_decay': 0.0},
        {'params': [p for n, p in list(model.attention.named_parameters()) if not any(nd in n for nd in no_decay)],
         'lr': lr*2, 'weight_decay':weight_decay},
        {'params': [p for n, p in list(model.attention.named_parameters()) if any(nd in n for nd in no_decay)],
         'lr': lr*2, 'weight_decay': 0.0},
        {'params': [p for n, p in list(model.label_embed.named_parameters()) if not any(nd in n for nd in no_decay)],
         'lr': lr*2, 'weight_decay':weight_decay},
        {'params': [p for n, p in list(model.label_embed.named_parameters()) if any(nd in n for nd in no_decay)],
         'lr': lr*2, 'weight_decay': 0.0}
        
    ]
    
#    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters2, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=3 *math.ceil(len(train_dataframe)/batch_size),
                                                num_training_steps=num_train_epochs*math.ceil(len(train_dataframe)/batch_size))
    
    print('-----start training!-----')
    best_f1=0.0
    patience = 0.0002
    patience_num = 10
    patience_cnt=0
    for epoch in range(num_train_epochs):
        '''
        if epoch==10:
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters2, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
            
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=3 *math.ceil(len(train_dataframe)/batch_size),
                                                        num_training_steps=(num_train_epochs-10)*math.ceil(len(train_dataframe)/batch_size))
        '''
        model.train()
        print('training epoch: {},total: {}'.format(epoch+1,num_train_epochs))
        train_loss=0
        for idx,batch_samples in enumerate(tqdm(train_loader)):
            for key in batch_samples.keys():
                batch_samples[key]=batch_samples[key].to(device)
            
            labels=batch_samples.pop('labels')
            
            output,embed=model(batch_samples)
            cl_fct=CLloss().to(device)
            if len(weight_list) == 0:
                loss_fct = torch.nn.BCEWithLogitsLoss().to(device)
                #第一项: logits.view(-1, self.model.config.num_labels)
                #第二项: labels.float().view(-1, self.model.config.num_labels)
                #其实就是logits和labels，然后计算二者的loss
                loss = loss_fct( output.view(-1, model.num_labels),
                                labels.float().view(-1, model.num_labels))
            else:
                #print("weights:")
                #print(weight_list)
                weights = torch.tensor(weight_list).to(device)
                loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight = weights).to(device)
                
                loss = loss_fct(output.view(-1, model.num_labels),
                                labels.float().view(-1, model.num_labels))
            loss=loss+0.1*cl_fct(embed,labels)
            train_loss+=loss.item()
            loss.backward()
        
            optimizer.step()
            scheduler.step()
            
            model.zero_grad()
            optimizer.zero_grad()
        print('train_loss: {}'.format(float(train_loss) / len(train_loader)))
        model.eval()
        x=torch.tensor([]).to(device)
        y=torch.tensor([]).to(device)
        
        with torch.no_grad():
            for idx,batch_samples in enumerate(tqdm(train_loader)):
                for key in batch_samples.keys():
                    batch_samples[key]=batch_samples[key].to(device)
            
                labels=batch_samples.pop('labels')
            
                _,embed=model(batch_samples)
                x=torch.cat((x,embed), dim=0)
                y=torch.cat((y,labels), dim=0)
                
            y=y.float()
        '''    
        metrics=run_eval(model,train_loader,label_info,value_classes,x,y)
        print('result on train set:')
        print(metrics)
        '''
        metrics=run_eval(model,valid_loader,label_info,value_classes,x,y)
        print('result on valid set:')
        print(metrics)
    
        if metrics['marco-avg-f1score']-best_f1>patience:
            best_f1=metrics['marco-avg-f1score']
            patience_cnt=0
            torch.save({'model': model.state_dict()}, os.path.join(model_dir, 'model.pth'))
            x=x.to(torch.device('cpu'))
            y=y.to(torch.device('cpu'))
            np.save(os.path.join(model_dir, 'X.npy'),x)
            np.save(os.path.join(model_dir, 'Y.npy'),y)
            x=x.to(device)  
            y=y.to(device)
            print('-------saved model!-------')
        else:
            patience_cnt+=1
            if patience_cnt>=patience_num and epoch>int(num_train_epochs/2):
                break
            
            
def predict_bert_model(dataframe, model_dir, labels, batch_size, add_label_flag):
    dataframe=dataframe.reset_index(drop=True)
    add_conclusion(dataframe)
    
    ds, value_classes = convert_to_dataset(dataframe, dataframe, labels)
    
    label_info=tokenizer(label_describe, truncation=True,padding=True)
    
    for key in label_info.keys():
        label_info[key]=torch.tensor(label_info[key],dtype=torch.int32)
        
    ds = ds.remove_columns(['labels'])
    
    x=np.load(os.path.join(model_dir, 'X.npy'))
    y=np.load(os.path.join(model_dir, 'Y.npy'))
    x=torch.tensor(x,dtype=torch.float32).to(device)
    y=torch.tensor(y,dtype=torch.float32).to(device)
    base_model=AutoModel.from_pretrained('roberta-large')
    model=MyModel(base_model, len(labels),label_info,freeze_bert=False)
    model=model.to(device)
    state_dict = torch.load(os.path.join(model_dir, 'model.pth'))
    model.load_state_dict(state_dict['model'])
    data_loader=DataLoader(ds['train'],batch_size=batch_size,shuffle=False,collate_fn=my_collate)
    model.eval()
    y_pred=torch.tensor([]).to(device)
    with torch.no_grad():
            
        for idx,batch_samples in enumerate(tqdm(data_loader)):
           for key in batch_samples.keys():
               batch_samples[key]=batch_samples[key].to(device)
           
           output,embed=model(batch_samples)
           KNNpred=get_KNNpred(embed,x,y)
           output=output.sigmoid()
           output=0.75*output+0.25*KNNpred
           y_pred=torch.cat((y_pred,output), dim=0)
        #print(y_pred)
        y_pred=y_pred.to(torch.device('cpu'))
        
        #print(y_pred)
        prediction = 1 * (y_pred>0.5) 
        prediction=prediction.numpy()
        #print(prediction)
        
        return prediction
'''      
train_dataframe = pd.read_csv("../test.tsv", encoding='utf-8', sep='\t', header=0)

labels=['Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement', 'Power: dominance', 'Power: resources', 'Face', 'Security: personal', 'Security: societal', 'Tradition', 'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring', 'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature', 'Universalism: tolerance', 'Universalism: objectivity']

train_bert_model(train_dataframe,
                    './',
                    labels,
                    batch_size=8,
                    weight_list=[],
                    test_dataframe=None,
                    num_train_epochs=10,
                    add_label_flag = False)

#predict_bert_model(train_dataframe, './', labels, batch_size=1, add_label_flag=False)
'''

