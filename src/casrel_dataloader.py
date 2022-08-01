# coding: utf-8

import json
from transformers import AlbertTokenizer, AutoTokenizer
from torch.utils.data import Dataset,DataLoader
import random
import torch
import re
from random import choice
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class dataprocess(Dataset):
    def __init__(self, data, embed_mode, max_seq_len, rel2idx):
        self.data = data
        self.len = max_seq_len
        self.rel2idx = rel2idx
        if embed_mode == "albert":
            pass
            self.tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v1")
        elif embed_mode == "bert_cased":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        elif embed_mode == "scibert":
            pass
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        words = self.data[idx][0]
        sent_str = " ".join(words)
        rc_labels = self.data[idx][1]
        
        if len(words) > self.len:
            words, rc_labels = self.truncate(self.len, words, rc_labels)

        word_to_bep = self.map_origin_word_to_bert(words)
        rc_labels = self.rc_label_transform(rc_labels, word_to_bep)

        tokenized_sent = self.tokenizer(sent_str)
        sent_ids, tokens = tokenized_sent["input_ids"], tokenized_sent.tokens()
        encoded_sent = self.encode_sub_rel_obj(sent_ids, rc_labels)

        return encoded_sent + (rc_labels, tokens)

    def encode_sub_rel_obj(self, sent_ids, rc_labels):
        sub_start = len(sent_ids) * [0]
        sub_end = len(sent_ids) * [0]
        sub_start_single = len(sent_ids) * [0]
        sub_end_single = len(sent_ids) * [0]

        n_rels = len(self.rel2idx.keys())
        relation_start = [[0 for _ in range(n_rels)] for _ in range(len(sent_ids))]
        relation_end = [[0 for _ in range(n_rels)] for _ in range(len(sent_ids))]
        s2ro_map = {}
        for i in range(0, len(rc_labels), 3):
            sub_pos = rc_labels[i]
            obj_pos = rc_labels[i+1]
            relation = rc_labels[i+2]

            relation_idx = self.rel2idx[relation]
            sub_start[sub_pos[0]] = 1
            sub_end[sub_pos[1]] = 1
            if sub_pos not in s2ro_map:
                s2ro_map[sub_pos] = []
            s2ro_map[sub_pos].append((obj_pos, relation_idx))

        if s2ro_map:
            sub_pos = choice(list(s2ro_map.keys()))
            sub_start_single[sub_pos[0]] = 1
            sub_end_single[sub_pos[1]] = 1
            for obj_pos, relation_idx in s2ro_map.get(sub_pos, []):
                relation_start[obj_pos[0]][relation_idx] = 1
                relation_end[obj_pos[1]][relation_idx] = 1
        return sent_ids, sub_start, sub_end, relation_start, relation_end, sub_start_single, sub_end_single

    def map_origin_word_to_bert(self, words):
        bep_dict = {}
        current_idx = 0
        for word_idx, word in enumerate(words):
            bert_word = self.tokenizer.tokenize(word)
            word_len = len(bert_word)
            bep_dict[word_idx] = [current_idx, current_idx + word_len - 1]
            current_idx = current_idx + word_len
        return bep_dict
    
    def rc_label_transform(self, rc_label, word_to_bert):
        new_rc_labels = []

        for i in range(0, len(rc_label), 3):
            # +1 for [CLS]
            sub_start_idx, sub_end_idx = rc_label[i][0], rc_label[i][1]
            obj_start_idx, obj_end_idx = rc_label[i+1][0], rc_label[i+1][1]
            if sub_start_idx>=0 and sub_end_idx>=0 and obj_start_idx>=0 and obj_end_idx>=0:
                sub_start, sub_end = word_to_bert[sub_start_idx][0] + 1, word_to_bert[sub_end_idx][1] + 1
                obj_start, obj_end = word_to_bert[rc_label[i+1][0]][0] + 1, word_to_bert[rc_label[i+1][1]][1] + 1
                new_rc_labels += [(sub_start, sub_end), (obj_start, obj_end), rc_label[i + 2]]

        return new_rc_labels
    
    def truncate(self, max_seq_len, words, rc_labels):
        truncated_words = words[:max_seq_len]
        truncated_rc_labels = []
        
        for i in range(0, len(rc_labels), 3):
            if rc_labels[i][1] < max_seq_len and rc_labels[i+1][1] < max_seq_len:
                truncated_rc_labels += [rc_labels[i], rc_labels[i+1], rc_labels[i+2]]

        return truncated_words, truncated_rc_labels

def lst_substr_match(txt, sub_txt):
    for i in range(0, len(txt) - len(sub_txt)+1):
        if txt[i:i+len(sub_txt)] == sub_txt:
            return i, i + len(sub_txt) - 1

    print("failed case")
    return -1, -1

def data_preprocess(data):
    processed = []
    for dic in data:
        text = re.sub(r'[^\w\s]','',dic['text']).split()
        rc_labels = []
        trips = dic['triple_list']
        for trip in trips:
            subj = lst_substr_match(text, re.sub(r'[^\w\s]', '', trip[0]).split())
            obj = lst_substr_match(text, re.sub(r'[^\w\s]', '', trip[2]).split())
            rel = trip[1]

            rc_labels+=[subj,obj,rel]

        processed += [(text,rc_labels)]
    return processed

def json_load(path, name):
    file = path + "/" + name
    with open(file,'r') as f:
        return json.load(f)

def gen_rc_labels(rc_list, l, rel2idx):
    labels = torch.FloatTensor(l, l, len(rel2idx)).fill_(0)
    for i in range(0, len(rc_list), 3):
        e1 = rc_list[i]
        e2 = rc_list[i + 1]
        r = rc_list[i + 2]
        labels[e1][e2][rel2idx[r]] = 1

    return labels
    
def mask_to_tensor(len_list, batch_size):
    token_len = max(len_list)
    tokens = torch.LongTensor(token_len, batch_size).fill_(0)
    for i, s in enumerate(len_list):
        tokens[:s, i] = 1

    return tokens

def tensorize_list_items(items, device, batch_first, padding_value):
    item_tensors = [torch.tensor(i).float().to(device) for i in items]
    padded_item_tensors = pad_sequence(item_tensors, batch_first, padding_value)
    return padded_item_tensors

def tensorize_multi_list_items(device, batch_first, padding_value, *args):
    return [tensorize_list_items(items, device, batch_first, padding_value) for items in args]

def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    sample, sub_start, sub_end, relation_start, relation_end, sub_start_single, sub_end_single, rc_labels, tokens = zip(*data)

    mask = [[1 if j < len(i) else 0 for j in range(len(sample[0]))] for i in sample]
    mask = torch.tensor(mask).long().to(device)

    batch_first = True
    padding_value = 0

    sample, sub_start, sub_end, relation_start, relation_end, sub_start_single, sub_end_single = tensorize_multi_list_items(device, batch_first, padding_value, sample, sub_start, sub_end, relation_start, relation_end, sub_start_single, sub_end_single)

    return sample, sub_start, sub_end, relation_start, relation_end, mask, sub_start_single, sub_end_single, rc_labels, tokens

def dataloader(args):
    data = json_load(args["path"], f'{args["mode"]}_triples.json')

    with open(args["path"] + "/rel2id.json", "r") as f:
        rel2idx = json.load(f)[1]

    data = data_preprocess(data)
    processed_dataset = dataprocess(data, args["embed_mode"], args["max_seq_len"], rel2idx)
    batch_data = DataLoader(dataset=processed_dataset, batch_size=args["batch_size"], shuffle=True, collate_fn=collate_fn)

    return batch_data
