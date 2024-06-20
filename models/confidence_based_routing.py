import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.data_utils import class_maps

class Confidence_Based_Routing(nn.Module):
    def __init__(self, view_models, mode):
        super().__init__()

        self.mode = mode

        # view classifiers
        self.wase_model = view_models[0]
        self.camus_model = view_models[1]
        self.mstr_model = view_models[2]
        self.stg_model = view_models[3]

    def forward(self, x):

        with torch.no_grad():
            wase_view_logit = self.wase_model(x)
            wase_view_softmax = F.softmax(wase_view_logit, dim=-1)
            wase_view_pred = torch.argmax(wase_view_softmax).cpu().numpy()

            camus_view_logit = self.camus_model(x)
            camus_view_softmax = F.softmax(camus_view_logit, dim=-1)
            camus_view_pred = torch.argmax(camus_view_softmax).cpu().numpy()

            mstr_view_logit = self.mstr_model(x)
            mstr_view_softmax = F.softmax(mstr_view_logit, dim=-1)
            mstr_view_pred = torch.argmax(mstr_view_softmax).cpu().numpy()

            stg_view_logit = self.stg_model(x)
            stg_view_softmax = F.softmax(stg_view_logit, dim=-1)
            stg_view_pred = torch.argmax(stg_view_softmax).cpu().numpy()

            if self.mode == 'logit':
                logit_max = torch.tensor([wase_view_logit.max(),
                                          camus_view_logit.max(),
                                          mstr_view_logit.max(),
                                          stg_view_logit.max()])
                route_pred = int(torch.argmax(logit_max).cpu().numpy())
            elif self.mode == 'softmax':
                softmax_max = torch.tensor([wase_view_softmax.max(),
                               camus_view_softmax.max(),
                               mstr_view_softmax.max(),
                               stg_view_softmax.max()])
                route_pred = int(torch.argmax(softmax_max).cpu().numpy())
            else:
                print('No mode selected')

            all_view_preds = [wase_view_pred, camus_view_pred, mstr_view_pred, stg_view_pred]
            final_view_pred = all_view_preds[route_pred]

            # need to convert labels to global labels here
            class_map = class_maps(route_pred)
            #print(final_view_pred)
            final_view_pred = class_map[int(final_view_pred)]

        return final_view_pred




