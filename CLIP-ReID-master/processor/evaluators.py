import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import logging
import os
from typing import Optional, Dict, List
class PCBStyleEvaluator:
    def __init__(self, num_query, max_rank=50, feat_norm=True):
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reset()
        
        self.cmc_configs = {
            'allshots': {
                'separate_camera_set': False,
                'single_gallery_shot': False,
                'first_match_break': False
            },
            'cuhk03': {
                'separate_camera_set': True,
                'single_gallery_shot': True,
                'first_match_break': False
            },
            'market1501': {
                'separate_camera_set': False,
                'single_gallery_shot': False,
                'first_match_break': True
            }
        }

    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []

    def update(self, output):

        feat, pid, camid = output

        if isinstance(feat, torch.Tensor):
            self.features.append(feat.cpu())
        else:
            self.features.append(torch.tensor(feat).float())
        

        if hasattr(pid, 'cpu'): 
            self.pids.extend(pid.cpu().numpy())
        elif isinstance(pid, (list, tuple, np.ndarray)):
            self.pids.extend(np.array(pid))
        else:
            self.pids.append(pid) 
        

        if hasattr(camid, 'cpu'):
            self.camids.extend(camid.cpu().numpy())
        elif isinstance(camid, (list, tuple, np.ndarray)):
            self.camids.extend(np.array(camid))
        else:
            self.camids.append(camid)

    def compute(self):
        features = torch.cat(self.features, dim=0)
        if self.feat_norm:
            features = F.normalize(features, p=2, dim=1)

   
        qf = features[:self.num_query]
        gf = features[self.num_query:]
        q_pids = np.asarray(self.pids[:self.num_query])
        g_pids = np.asarray(self.pids[self.num_query:])
        q_camids = np.asarray(self.camids[:self.num_query])
        g_camids = np.asarray(self.camids[self.num_query:])

  
          distmat = 1 - torch.mm(qf, gf.T).numpy() 


        cmc_scores = {}
        mAP = self._compute_mAP(distmat, q_pids, g_pids, q_camids, g_camids)
        
        for name, params in self.cmc_configs.items():
            cmc_scores[name] = self._compute_cmc(
                distmat, q_pids, g_pids, q_camids, g_camids, **params
            )


        return (
            cmc_scores['allshots'], 
            mAP,                    
            cmc_scores['cuhk03'],    
            cmc_scores['market1501'], 
            [], [], []               
        )

    def _compute_mAP(self, distmat, q_pids, g_pids, q_camids, g_camids):
        """计算mAP"""
        num_q, num_g = distmat.shape
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        aps = []
        for q_idx in range(num_q):

            order = indices[q_idx]
            keep = (g_camids[order] != q_camids[q_idx]) | (g_pids[order] != q_pids[q_idx])
            matches_q = matches[q_idx][keep]

            if np.sum(matches_q) == 0:
                continue

            rel = matches_q
            prec = rel.cumsum() / (np.arange(len(rel)) + 1)
            ap = np.sum(prec * rel) / np.sum(rel)
            aps.append(ap)

        return np.mean(aps) if aps else 0

    def _compute_cmc(self, distmat, q_pids, g_pids, q_camids, g_camids,
               separate_camera_set, single_gallery_shot, first_match_break):

        num_q, num_g = distmat.shape
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
        ret = np.zeros(self.max_rank)
        num_valid = 0
    
        for q_idx in range(num_q):

            if separate_camera_set:
                order = indices[q_idx]
                keep = (g_camids[order] != q_camids[q_idx])
                matches_q = matches[q_idx][keep]
            else:

                order = indices[q_idx]
                keep = (g_camids[order] != q_camids[q_idx]) | (g_pids[order] != q_pids[q_idx])
                matches_q = matches[q_idx][keep]
    
    
            if q_idx == 0:  
                print(f"Sample {q_idx}: Matches: {matches_q[:10]}... (total matches: {np.sum(matches_q)})")
    
            if first_match_break:
                tmp_cmc = matches_q[:self.max_rank].cumsum()
                tmp_cmc[tmp_cmc > 1] = 1
            else:
                tmp_cmc = matches_q.cumsum()
                tmp_cmc[tmp_cmc > 1] = 1
                tmp_cmc = tmp_cmc[:self.max_rank]
            
            ret += tmp_cmc
            num_valid += 1

        return ret / num_valid if num_valid > 0 else ret

    def print_results(self, cmc, mAP, cmc_cuhk03=None, cmc_market1501=None):
        """打印PCB风格结果"""
        print('Mean AP: {:.1%}'.format(mAP))
        print('CMC Scores{:>12}{:>12}{:>12}'.format(
            'allshots', 'cuhk03', 'market1501'))
        
        ranks = [1, 5, 10]
        for rank in ranks:
            print('  top-{:<11}{:12.1%}{:12.1%}{:12.1%}'.format(
                rank,
                cmc[rank-1],
                cmc_cuhk03[rank-1] if cmc_cuhk03 is not None else 0,
                cmc_market1501[rank-1] if cmc_market1501 is not None else 0
            ))