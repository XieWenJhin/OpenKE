import openke
from openke.config import Trainer, Tester
from openke.module.model import SimplE, ComplEx
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

import torch
import torch.nn.functional as F
import pandas as pd
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    if args.model == "complex":
        model = ComplEx(
            ent_tot = 5101080,
            rel_tot = 3,
            dim = 100
        )
        model.load_checkpoint('./checkpoint/imdb_complEx.ckpt')
    elif args.model == 'simple':
        model = SimplE(
            ent_tot = 5101080,
            rel_tot = 3,
            dim = 200
        )
        model.load_checkpoint('./checkpoint/imdb_simple.ckpt')
    model = model.cuda()
    pos = pd.read_csv('/home/LAB/xiewj/neural-mining/datasets/imdb/test_edges.csv')
    neg_fsrc = pd.read_csv('/home/LAB/xiewj/neural-mining/datasets/imdb/test_neg_per_src/test_neg_5.csv')
    neg_fdst = pd.read_csv('/home/LAB/xiewj/neural-mining/datasets/imdb/test_neg_per_dst/test_neg_5.csv')
    
    pos_tri = dict()
    pos_tri['batch_h'] = torch.from_numpy(pos['source_id:int'].values).cuda()
    pos_tri['batch_t'] = torch.from_numpy(pos['target_id:int'].values).cuda()
    pos_tri['batch_r'] = torch.from_numpy(pos['label_id:int'].values).cuda()

    neg_fsrc_tri = dict()
    neg_fsrc_tri['batch_h'] = torch.from_numpy(neg_fsrc['source_id:int'].values).cuda()
    neg_fsrc_tri['batch_t'] = torch.from_numpy(neg_fsrc['target_id:int'].values).cuda()
    neg_fsrc_tri['batch_r'] = torch.from_numpy(neg_fsrc['label_id:int'].values).cuda()

    neg_fdst_tri = dict()
    neg_fdst_tri['batch_h'] = torch.from_numpy(neg_fdst['source_id:int'].values).cuda()
    neg_fdst_tri['batch_t'] = torch.from_numpy(neg_fdst['target_id:int'].values).cuda()
    neg_fdst_tri['batch_r'] = torch.from_numpy(neg_fdst['label_id:int'].values).cuda()

    pos_score = -model.predict(pos_tri)
    neg_fsrc_score = -model.predict(neg_fsrc_tri)
    neg_fdst_score = -model.predict(neg_fdst_tri)
    num_pos = pos_score.size
    num_neg = neg_fsrc_score.size

    pos_score = torch.from_numpy(pos_score)
    pos_socre = F.softplus(pos_score)

    neg_fsrc_score = torch.from_numpy(neg_fsrc_score)
    neg_fsrc_socre = F.softplus(neg_fsrc_score)

    neg_fdst_score = torch.from_numpy(neg_fdst_score)
    neg_fdst_socre = F.softplus(neg_fdst_score)
    
    t = dict()
    t['t'] = 0
    t['f1'] = 0
    for i in range(-100, 0):
        alpha = i / 10;
        TP = (pos_score > alpha).sum().item()
        FN = num_pos - TP
        FP = (neg_fsrc_score > alpha).sum().item()
        TN = num_neg - FP

    # print(f'TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}')
        prec = TP / (TP + FP)
        recall = TP / (TP + FN)
        fsrc_f1 = 2 * prec * recall / (prec + recall)
        
        FP = (neg_fdst_score > alpha).sum().item()
        TN = num_neg - FP

        prec = TP / (TP + FP)
        recall = TP / (TP + FN)
        fdst_f1 = 2 * prec * recall / (prec + recall)

        f1 = (fsrc_f1 + fdst_f1) / 2
        if f1 > t['f1']:
            t['t'] = alpha
            t['lf1'] = fsrc_f1
            t['rf1'] = fdst_f1
            t['f1'] = f1
    print(f"Threshold: {t['t']}, Fixed Src: {t['lf1']}, Fixed Dst: {t['rf1']}, Best F1: {t['f1']}")
    