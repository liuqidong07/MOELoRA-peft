# -*- encoding: utf-8 -*-
# here put the import lib
import json
import jsonlines
from tqdm import tqdm
import numpy as np
import os
from post_generate_process import process_generated_results
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from rouge import Rouge
import jieba



def list2dict(data_list):

    data_dict = {}
    for meta_data in data_list:
        if meta_data["sample_id"] in data_dict.keys():
            raise ValueError("Duplicated Sample ID !")
        data_dict[meta_data["sample_id"]] = meta_data["answer"]
    
    return data_dict


def calculate_ner_f1(true_list, pred_list):
    """Calculate F1 score for NER"""
    TP, FP, FN = 0, 0, 0
    for i in tqdm(range(len(true_list))):
        for meta_true in true_list[i]:
            if meta_true in pred_list[i]:
                TP += 1
            else:
                FN += 1
        for meta_pred in pred_list[i]:
            if meta_pred not in true_list[i]:
                FP += 1
                
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


def calculate_score(task, pred_path, truth_path, post_process=None):

    pred = process_generated_results(pred_path)[task]
    pred = list2dict(pred)
    truth = process_generated_results(truth_path)[task]
    truth = list2dict(truth)
    assert len(pred) == len(truth)

    if post_process:
        post_process(pred)  # post process the predictions

    pred_data, true_data = [], []

    for sample_id in truth.keys():
        
        pred_data.append(pred[sample_id])
        true_data.append(truth[sample_id])
    
    #print(pred_data)
    if (task == "CHIP-CTC") | (task == "KUAKE-QTR") | (task == "CHIP-STS") | \
       (task == "KUAKE-QQR") | (task == "KUAKE-IR"):    # micro
        
        pred_data = np.array(pred_data)
        true_data = np.array(true_data)

        _, _, score1, _ = precision_recall_fscore_support(y_pred=pred_data, y_true=true_data, labels=np.unique(true_data))
        _, _, score2, _ = precision_recall_fscore_support(y_pred=pred_data, y_true=true_data, average="micro")

        return score2, np.unique(true_data), score1  # detail results, average_results

    if (task == "CHIP-CTC") | (task == "KUAKE-QIC") | (task == "IMCS-V2-DAC"):  # macro
        
        pred_data = np.array(pred_data)
        true_data = np.array(true_data)

        _, _, score1, _ = precision_recall_fscore_support(y_pred=pred_data, y_true=true_data, labels=np.unique(true_data))
        _, _, score2, _ = precision_recall_fscore_support(y_pred=pred_data, y_true=true_data, average="macro")

        return score2, np.unique(true_data), score1  # detail results, average_results

    elif (task == "CMeEE-V2") | (task == "IMCS-V2-NER") | (task == "CMeIE") | \
         (task == "CHIP-CDEE") | (task == "IMCS-V2-SR") | (task == "CHIP-MDCFNPC") | \
         (task == "CHIP-CDN"):
        
        f1, precision, recall = calculate_ner_f1(true_data, pred_data)
        
        return f1, precision, recall
    
    elif (task == "MedDG"):
        
        rouger = Rouge()

        f1, precision, recall = 0, 0, 0
        for i in range(len(pred_data)):
            
            if not pred_data[i]:
                pred_data[i] = "-"
            rougel = rouger.get_scores(" ".join(list(jieba.cut(pred_data[i]))), " ".join(list(jieba.cut(true_data[i]))))[0]["rouge-l"]
            f1 += rougel["f"] / len(pred_data)
            precision += rougel["p"] / len(pred_data)
            recall += rougel["r"] / len(pred_data)
        
        return f1, precision, recall
    
    elif (task == "IMCS-V2-MRG"):

        rouger = Rouge()

        f1, precision, recall = 0, 0, 0
        for i in range(len(pred_data)):
            
            per_f1, per_precision, per_recall = 0, 0, 0
            for key in true_data[i].keys():
                if key in pred_data[i].keys():
                    rougel = rouger.get_scores("-" + " ".join(list(jieba.cut(pred_data[i][key]))), 
                                               "-" + " ".join(list(jieba.cut(true_data[i][key]))))[0]["rouge-l"]
                else:
                    rougel = {"f":0, "p":0, "r": 0}
                per_f1 += rougel["f"] / len(true_data[i].keys())
                per_precision += rougel["p"] / len(true_data[i].keys())
                per_recall += rougel["r"] / len(true_data[i].keys())

            f1 += per_f1 / len(pred_data)
            precision += per_precision / len(pred_data)
            recall += per_recall / len(pred_data)
        
        return f1, precision, recall
    
    else:

        raise ValueError("No such task %s" % task)


def process_CTC(pred):
    for sample_id in pred.keys():
        if "(医生检测)" in pred[sample_id]:
            pred[sample_id] = pred[sample_id].replace("(医生检测)", "(医生检测）")


