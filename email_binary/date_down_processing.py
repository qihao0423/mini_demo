import pandas as pd
import re
import numpy as np

positive = './data/rt-polarity.pos'
negative = './data/rt-polarity.neg'

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_lables():
    positive_txt = [s.strip() for s in list(open(positive,'r',encoding='utf-8').readlines())]
    negative_txt = [s.strip() for s in list(open(negative,'r',encoding='utf-8').readlines())]

    a = positive_txt + negative_txt
    data_x = [clean_str(s) for s in a]
    positive_lab = [[0,1] for i in positive_txt]
    negative_lab = [[1,0] for i in negative_txt]
    data_y = np.concatenate([positive_lab,negative_lab],0)

    return data_x,data_y
































