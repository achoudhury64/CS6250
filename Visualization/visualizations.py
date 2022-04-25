import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('white')
sns.set_context('talk', font_scale = 1)
import os

import math
from transformers import BertTokenizer

startingDir = os.getcwd() # save our current directory
print(startingDir)
os.chdir('../') # moves the current working directory up one

from readmissions_model import BertForSequenceClassification
import readmissions_model

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

os.chdir(startingDir)
bert_config = readmissions_model.BertConfig.from_json_file('../model/early_readmission/bert_config.json')
model = BertForSequenceClassification(bert_config, 1)

dicts = model.load_state_dict(torch.load('../model/early_readmission/pytorch_model.bin', map_location='cpu'))

def transpose_for_scores(config, x):
    new_x_shape = x.size()[:-1] + (config.num_attention_heads, int(config.hidden_size / config.num_attention_heads))
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)


def get_attention_scores(model, i, text):
    tokenized = tokenizer.tokenize(text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized)

    segment_ids = [0] * len(indexed_tokens)
    t_tensor = torch.tensor([indexed_tokens])
    s_ids = torch.tensor([segment_ids])

    outputs_query = []
    outputs_key = []

    def hook_query(module, input, output):
        # print ('in query')
        outputs_query.append(output)

    def hook_key(module, input, output):
        # print ('in key')
        outputs_key.append(output)

    model.bert.encoder.layer[i].attention.self.query.register_forward_hook(hook_query)
    model.bert.encoder.layer[i].attention.self.key.register_forward_hook(hook_key)
    l = model(t_tensor, s_ids)

    query_layer = transpose_for_scores(bert_config, outputs_query[0])
    key_layer = transpose_for_scores(bert_config, outputs_key[0])

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(int(bert_config.hidden_size / bert_config.num_attention_heads))
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    return attention_probs, tokenized

text=' he has experienced acute on chronic diastolic heart failure in the setting of volume overload due to his sepsis.'
x,tokens=get_attention_scores(model,0,text)
map1=np.asarray(x[0][1].detach().numpy())

f, ax = plt.subplots(figsize=(6,8))
sns.heatmap(map1, annot=False, fmt="f", ax=ax, xticklabels = False, yticklabels = False, vmax=0.4, cmap='Reds', cbar_kws={'label':'Attention Weight', 'orientation':'horizontal'}, rasterized = True)

plt.clf()

f=plt.figure(figsize=(10,10))
ax = f.add_subplot(1,1,1)
i=ax.imshow(map1,interpolation='nearest',cmap='Reds')

ax.set_yticks(range(len(tokens)))
ax.set_yticklabels(tokens)

ax.set_xticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=90)

ax.set_xlabel('key')
ax.set_ylabel('query')

ax.grid(linewidth = 0.8)