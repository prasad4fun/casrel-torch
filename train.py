import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import Casrel
from src.casrel_dataloader import dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_num_threads(6)

def get_loss(pred, gold, mask):
    pred = pred.squeeze(-1)
    loss = F.binary_cross_entropy(pred, gold.float(), reduction='none')
    if loss.shape != mask.shape:
        mask = mask.unsqueeze(-1)
    loss = torch.sum(loss*mask)/torch.sum(mask)
    return loss
  
  if __name__ == '__main__':
    config = {"mode": "train", "embed_mode": "bert_cased", "path": "data/NYT", 'batch_size': 64, 'epoch': 50, "max_seq_len": 128, "rel_num": 24}

    train_batch = dataloader(config)
    model = Casrel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
    loss_recorder = 0
    for epoch_index in range(config['epoch']):
        time_start = time.perf_counter()
        for batch_index, (sample, sub_start, sub_end, relation_start, relation_end, mask, sub_start_single, sub_end_single, gold_triples, tokens) in enumerate(iter(train_batch)):
            batch_data = {"token_ids": sample,
                          "mask": mask,
                          "sub_start": sub_start_single,
                          "sub_end": sub_end_single}

            pred_sub_start, pred_sub_end, pred_obj_start, pred_obj_end = model(batch_data)
            sub_start_loss = get_loss(pred_sub_start, sub_start, mask)
            sub_end_loss = get_loss(pred_sub_end, sub_end, mask)
            obj_start_loss = get_loss(pred_obj_start, relation_start, mask)
            obj_end_loss = get_loss(pred_obj_end, relation_end, mask)
            loss = (sub_start_loss + sub_end_loss) + (obj_start_loss + obj_end_loss)
            loss_recorder += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch: %d batch: %d loss: %f"% (epoch_index, batch_index, loss))
            if(batch_index%100 == 99):
                print(loss_recorder)
                loss_recorder = 0
        time_end = time.perf_counter()
        torch.save(model.state_dict(), 'params.pkl')
        print("successfully saved! time used = %fs."% (time_end-time_start))
