import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class MSP(nn.Module):
    def __init__(self, view_models):
        super().__init__()

        # view classifiers.
        # b=wase, n1=camus, n2=mstr, n3=stg
        self.b_model = view_models[0]
        for param in self.b_model.parameters():
            param.requires_grad = False
        self.b_model.eval()
        self.n1_model = view_models[1]
        for param in self.n1_model.parameters():
            param.requires_grad = False
        self.n1_model.eval()
        self.n2_model = view_models[2]
        for param in self.n2_model.parameters():
            param.requires_grad = False
        self.n2_model.eval()
        self.n3_model = view_models[3]
        for param in self.n3_model.parameters():
            param.requires_grad = False
        self.n3_model.eval()

        b_classes = 8
        n1_classes = 2
        n2_classes = 11
        n3_classes = 4

    def forward(self, x):
        with torch.no_grad():
            b_logits = self.b_model(x)
            n1_logits = self.n1_model(x)
            n2_logits = self.n2_model(x)
            n3_logits = self.n3_model(x)

        b_prob = F.softmax(b_logits, dim=-1)
        n1_prob = F.softmax(n1_logits, dim=-1)
        n2_prob = F.softmax(n2_logits, dim=-1)
        n3_prob = F.softmax(n3_logits, dim=-1)

        # non overlapping classes
        pred_a2c_con = n3_prob[:, 0]
        pred_a3c_con = n3_prob[:, 1]
        pred_a4c_con = n3_prob[:, 2]
        pred_plax_con = n3_prob[:, 3]
        pred_rv = n2_prob[:, 8]
        pred_sc = n2_prob[:, 9]
        pred_scivc = n2_prob[:, 10]

        #print(torch.stack([b_logits[:, 0], n1_logits[:, 0], n2_logits[:, 0]], dim=-1).shape)
        #print(torch.max(torch.stack([b_logits[:, 0], n1_logits[:, 0], n2_logits[:, 0]], dim=-1), dim=-1)[0])

        # maxpool overlapping classes
        pred_a2c = torch.max(torch.stack([b_prob[:, 0], n1_prob[:, 0], n2_prob[:, 0]], dim=-1), dim=-1)[0]
        pred_a3c = torch.max(torch.stack([b_prob[:, 1], n2_prob[:, 1]], dim=-1), dim=-1)[0]
        pred_a4c = torch.max(torch.stack([b_prob[:, 2], n1_prob[:, 1], n2_prob[:, 2]], dim=-1), dim=-1)[0]
        pred_a5c = torch.max(torch.stack([b_prob[:, 3], n2_prob[:, 3]], dim=-1), dim=-1)[0]
        pred_plax = torch.max(torch.stack([b_prob[:, 4], n2_prob[:, 4]], dim=-1), dim=-1)[0]
        pred_plaxav = torch.max(torch.stack([b_prob[:, 5], n2_prob[:, 5]], dim=-1), dim=-1)[0]
        pred_psaxav = torch.max(torch.stack([b_prob[:, 6], n2_prob[:, 6]], dim=-1), dim=-1)[0]
        pred_psaxpm = torch.max(torch.stack([b_prob[:, 7], n2_prob[:, 7]], dim=-1), dim=-1)[0]

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

        return combined_logits




