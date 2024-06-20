import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Score_Fusion(nn.Module):
    def __init__(self):
        super().__init__()

        b_classes = 8
        n1_classes = 2
        n2_classes = 11
        n3_classes = 4

        self.lin_n1_logits_b_features = nn.Linear(512, n1_classes)
        self.lin_n2_logits_b_features = nn.Linear(512, n2_classes)
        self.lin_n3_logits_b_features = nn.Linear(512, n3_classes)

        self.lin_b_logits_n1_features = nn.Linear(512, b_classes)
        self.lin_n2_logits_n1_features = nn.Linear(512, n2_classes)
        self.lin_n3_logits_n1_features = nn.Linear(512, n3_classes)

        self.lin_b_logits_n2_features = nn.Linear(512, b_classes)
        self.lin_n1_logits_n2_features = nn.Linear(512, n1_classes)
        self.lin_n3_logits_n2_features = nn.Linear(512, n3_classes)

        self.lin_b_logits_n3_features = nn.Linear(512, b_classes)
        self.lin_n1_logits_n3_features = nn.Linear(512, n1_classes)
        self.lin_n2_logits_n3_features = nn.Linear(512, n2_classes)

    def forward(self, features, logits):

        #print(nmd.shape, features.shape, logits.shape)

        b_features, n1_features, n2_features, n3_features = features[:,0,:], features[:,1,:], features[:,2,:], features[:,3,:]
        b_logits_b_features, n1_logits_n1_features, n2_logits_n2_features, n3_logits_n3_features = logits[:,:8], logits[:,8:10], logits[:,10:21], logits[:,21:]

        b_logits_n1_features = self.lin_b_logits_n1_features(n1_features)
        b_logits_n2_features = self.lin_b_logits_n2_features(n2_features)
        b_logits_n3_features = self.lin_b_logits_n3_features(n3_features)
        #print(b_logits_b_features)
        b_stack = torch.stack([b_logits_b_features,
                             b_logits_n1_features,
                             b_logits_n2_features,
                             b_logits_n3_features],dim=1)
        b_attn = b_stack
        b_logits = b_attn.sum(dim=1)

        n1_logits_b_features = self.lin_n1_logits_b_features(b_features)
        n1_logits_n2_features = self.lin_n1_logits_n2_features(n2_features)
        n1_logits_n3_features = self.lin_n1_logits_n3_features(n3_features)
        n1_stack = torch.stack([n1_logits_b_features,
                               n1_logits_n1_features,
                               n1_logits_n2_features,
                               n1_logits_n3_features], dim=1)
        n1_attn = n1_stack
        n1_logits = n1_attn.sum(dim=1)

        n2_logits_b_features = self.lin_n2_logits_b_features(b_features)
        n2_logits_n1_features = self.lin_n2_logits_n1_features(n1_features)
        n2_logits_n3_features = self.lin_n2_logits_n3_features(n3_features)
        n2_stack = torch.stack([n2_logits_b_features,
                                n2_logits_n1_features,
                                n2_logits_n2_features,
                                n2_logits_n3_features], dim=1)
        n2_attn = n2_stack
        n2_logits = n2_attn.sum(dim=1)

        n3_logits_b_features = self.lin_n3_logits_b_features(b_features)
        n3_logits_n1_features = self.lin_n3_logits_n1_features(n1_features)
        n3_logits_n2_features = self.lin_n3_logits_n2_features(n2_features)
        n3_stack = torch.stack([n3_logits_b_features,
                                n3_logits_n1_features,
                                n3_logits_n2_features,
                                n3_logits_n3_features], dim=1)
        n3_attn = n3_stack
        n3_logits = n3_attn.sum(dim=1)

        # non overlapping classes
        pred_a2c_con = n3_logits[:, 0]
        pred_a3c_con = n3_logits[:, 1]
        pred_a4c_con = n3_logits[:, 2]
        pred_plax_con = n3_logits[:, 3]
        pred_rv = n2_logits[:, 8]
        pred_sc = n2_logits[:, 9]
        pred_scivc = n2_logits[:, 10]

        # maxpool overlapping classes
        pred_a2c = torch.max(torch.stack([b_logits[:, 0], n1_logits[:, 0], n2_logits[:, 0]], dim=-1), dim=-1)[0]
        pred_a3c = torch.max(torch.stack([b_logits[:, 1], n2_logits[:, 1]], dim=-1), dim=-1)[0]
        pred_a4c = torch.max(torch.stack([b_logits[:, 2], n1_logits[:, 1], n2_logits[:, 2]], dim=-1), dim=-1)[0]
        pred_a5c = torch.max(torch.stack([b_logits[:, 3], n2_logits[:, 3]], dim=-1), dim=-1)[0]
        pred_plax = torch.max(torch.stack([b_logits[:, 4], n2_logits[:, 4]], dim=-1), dim=-1)[0]
        pred_plaxav = torch.max(torch.stack([b_logits[:, 5], n2_logits[:, 5]], dim=-1), dim=-1)[0]
        pred_psaxav = torch.max(torch.stack([b_logits[:, 6], n2_logits[:, 6]], dim=-1), dim=-1)[0]
        pred_psaxpm = torch.max(torch.stack([b_logits[:, 7], n2_logits[:, 7]], dim=-1), dim=-1)[0]

        # stack all logits
        combined_logits = torch.stack([pred_a2c_con,
                                       pred_a3c_con,
                                       pred_a4c_con,
                                       pred_plax_con,
                                       pred_a2c,
                                       pred_a3c,
                                       pred_a4c,
                                       pred_a5c,
                                       pred_plax,
                                       pred_plaxav,
                                       pred_psaxav,
                                       pred_psaxpm,
                                       pred_rv,
                                       pred_sc,
                                       pred_scivc], dim=-1)


        return combined_logits, [combined_logits.cpu(), b_stack.cpu(), n1_stack.cpu(), n2_stack.cpu(), n3_stack.cpu()]




