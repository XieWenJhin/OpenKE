import openke
from openke.config import Trainer, Tester
from openke.module.model import SimplE
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

import torch
import pandas as pd

if __name__ == "__main__":
    simple = SimplE(
        ent_tot = 205783,
        rel_tot = 17,
        dim = 200
    )
    simple = simple.cuda()
    simple.load_checkpoint('./checkpoint/dblp_simple.ckpt')
    pos = pd.read_csv('/home/LAB/xiewj/neural-mining/datasets/dblp/test_edges.csv')
    neg = pd.read_csv('/home/LAB/xiewj/neural-mining/datasets/dblp/test_neg_per_src/test_neg_100.csv')
    
    pos_tri = dict()
    pos_tri['batch_h'] = torch.from_numpy(pos['source_id:int'].values).cuda()
    pos_tri['batch_t'] = torch.from_numpy(pos['target_id:int'].values).cuda()
    pos_tri['batch_r'] = torch.from_numpy(pos['label_id:int'].values).cuda()

    neg_tri = dict()
    neg_tri['batch_h'] = torch.from_numpy(neg['source_id:int'].values).cuda()
    neg_tri['batch_t'] = torch.from_numpy(neg['target_id:int'].values).cuda()
    neg_tri['batch_r'] = torch.from_numpy(neg['label_id:int'].values).cuda()

    score = simple.predict(pos_tri)
    TP = (torch.from_numpy(score).sigmoid() > 0.5).sum()
    FN = score.size - TP

    score = simple.predict(neg_tri)
    FP = (torch.from_numpy(score).sigmoid() > 0.5).sum()
    TN = score.size - TP

    print(f'TP: {TP}, FP: {FN}, FP: {FP}, TN: {TN}')
    prec = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * prec * recall / (prec + recall)
    print(f1)
    