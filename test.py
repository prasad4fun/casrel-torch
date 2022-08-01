import time

import torch
import torch.nn.functional as F
from src.model import Casrel
from torch.utils.data import DataLoader
from src.casrel_dataloader import dataloader
import numpy as np
import json, os

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(6)

def get_list(start, end, text, h_bar=0.5, t_bar=0.5):
    res = []
    start, end = start[: 512], end[: 512]
    start_idxs, end_idxs = [], []

    for idx in range(len(start)):
        if (start[idx] > h_bar):
            start_idxs.append(idx)
        if (end[idx] > t_bar):
            end_idxs.append(idx)
    for start_idx in start_idxs:
        for end_idx in end_idxs:
            if (end_idx >= start_idx):
                entry = [text[start_idx: end_idx+1], start_idx, end_idx]
                res.append(entry)
                # break
    return res

def get_triples(subjects, encoded_text, model, config):
    spo_set = set()

    def map_sub_start_end(subjects):
        repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
        sub_start_mapping = torch.zeros(len(subjects), 1, encoded_text.shape[1]).to(device)
        sub_end_mapping = torch.zeros(len(subjects), 1, encoded_text.shape[1]).to(device)
        for idx, sub in enumerate(subjects):
            sub_start_mapping[idx][0][sub[1]] = 1
            sub_end_mapping[idx][0][sub[2]] = 1
        return sub_start_mapping, sub_end_mapping, repeated_encoded_text

    sub_start_mapping, sub_end_mapping, repeated_encoded_text = map_sub_start_end(subjects)
    pred_obj_start, pred_obj_end = model.get_objs_for_specific_sub(sub_start_mapping, sub_end_mapping, repeated_encoded_text)

    for idx, sub in enumerate(subjects):
        sub_txt = "".join(sub[0])
        obj_start, obj_end = pred_obj_start[idx].transpose(0, 1), pred_obj_end[idx].transpose(0, 1)
        for i in range(config['rel_num']):
            obj_list = get_list(obj_start[i], obj_end[i], tokens[0])

            for obj in obj_list:
                obj_txt = "".join(obj[0])
                spo_item = (sub_txt, id2rel[str(i)], obj_txt)
                spo_set.add(spo_item)
            # if not spo_set:
            #     spo_set.add((sub_txt, "", ""))

    return spo_set


def transform_gold_triple(gold_triple):
    gold_sub_start, gold_sub_end = gold_triple[0]
    gold_obj_start, gold_obj_end = gold_triple[1]

    gold_sub = tokens[0][gold_sub_start:gold_sub_end+1]
    gold_obj = tokens[0][gold_obj_start:gold_obj_end+1]
    gold_rel = gold_triple[2]

    return [gold_sub, gold_obj, gold_rel]

def index_to_text_triples(gold_triples):
    gold_triples_transformed = []

    for sent_triples in gold_triples:
        for i in range(0, len(sent_triples), 3):
            res = transform_gold_triple(sent_triples[i:i + 3])
            gold_triples_transformed.extend(res)

    return gold_triples_transformed



if __name__ == '__main__':
    config = {"mode": "test", "embed_mode": "bert_cased", "path": "data/NYT", 'batch_size': 1, "max_seq_len": 128, "rel_num": 24}
    id2rel = json.load(open(os.path.join(config["path"], 'rel2id.json')))[0]
    h_bar, t_bar = 0.5, 0.5

    test_batch = dataloader(config)
    model = Casrel(config).to(device)
    model.load_state_dict(torch.load('params.pkl'))


    for batch_index, (sample, sub_start, sub_end, relation_start, relation_end, mask, sub_start_single, sub_end_single, gold_triples, tokens) in enumerate(iter(test_batch)):
        with torch.no_grad():
            gold_triples_transformed = index_to_text_triples(gold_triples)

            encoded_text = model.get_encoded_text(sample, mask)
            pred_sub_heads, pred_sub_tails = model.get_subs(encoded_text)
            subjects = get_list(pred_sub_heads.squeeze(), pred_sub_tails.squeeze(), tokens[0])
            spo_set = get_triples(subjects, encoded_text, model, config)

            res = {}
            res['text'] = ' '.join(tokens[0])
            res['pred_spo'] = spo_set
            res["gold_spo"] = gold_triples_transformed
            if res['pred_spo']:
                print(res)
