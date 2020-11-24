import re
import zipfile
import pandas as pd

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

with zipfile.ZipFile('data/open.zip', 'r') as zip_ref:
    zip_ref.extractall('data/')
    
class TextDataset(Dataset):
    def __init__(self,separation=True):
        self.df = pd.read_csv('data/train.csv', index_col='index')
        X,Y = [],[]
        for idx in self.df.index:
            texts = self.df.text[idx].split('. ')
            author = self.df.author[idx]
            for text in texts:
                text = re.sub(r'[^A-Za-z ]', '', text)
                X.append(text.lower())
                Y.append(author)
                
        self.X = X
        self.Y = Y
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.separation = separation
    def __len__(self):
        if self.separation:
            return len(self.X)
        else:
            return len(self.df)
    def __getitem__(self,idx):
        if self.separation:
            x = self.tokenizer(self.X[idx])
            return torch.tensor(x['input_ids']),torch.tensor(x['attention_mask']), torch.tensor(self.Y[idx])
        else:
            x = self.tokenizer(self.df.text[idx])
            return torch.tensor(x['input_ids']),torch.tensor(x['attention_mask']), torch.tensor(self.df.author[idx])