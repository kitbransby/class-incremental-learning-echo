import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys


class Neural_Mean_Discrepancy(nn.Module):
    def __init__(self, view_models, layer_names, nmf):
        super().__init__()

        self.layer_names = layer_names
        self.activation = {}

        # neural mean feature
        self.wase_nmf = nmf[0]
        self.camus_nmf = nmf[1]
        self.mstr_nmf = nmf[2]
        self.stg_nmf = nmf[3]

        # view classifiers
        self.wase_model = view_models[0]
        self.camus_model = view_models[1]
        self.mstr_model = view_models[2]
        self.stg_model = view_models[3]

        # register the hooks using the requested layer names
        self.register_activations(self.wase_model)
        self.register_activations(self.camus_model)
        self.register_activations(self.mstr_model)
        self.register_activations(self.stg_model)

    def get_activations(self, name):
        # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def register_activations(self, cls_model):
        cls_model.eval()
        # register a forward hook for every requested layer name
        for name, layer in cls_model.named_modules():
            if name in self.layer_names:
                layer.register_forward_hook(self.get_activations(name))

    def forward(self, x):

        with torch.no_grad():
            wase_feat, wase_logits, wase_activations = self.calc_activations(x, self.wase_model, self.wase_nmf)
            camus_feat, camus_logits, camus_activations = self.calc_activations(x, self.camus_model, self.camus_nmf)
            mstr_feat, mstr_logits, mstr_activations = self.calc_activations(x, self.mstr_model, self.mstr_nmf)
            stg_feat, stg_logits, stg_activations = self.calc_activations(x, self.stg_model, self.stg_nmf)

            activation_vec = torch.cat([wase_activations, camus_activations, mstr_activations, stg_activations], dim=-1)
            features = [wase_feat, camus_feat, mstr_feat, stg_feat]
            logits = [wase_logits, camus_logits, mstr_logits, stg_logits]

        return activation_vec, features, logits

    def calc_activations(self, x, cls_model, nmf):

        # pass single input through model
        features, logits = cls_model.features_and_logits(x)
        #print('calc act: features, logits ', features.shape, logits.shape)

        # calculate activation vector

        activations = torch.cat([self.activation[layer_name].mean(dim=[2, 3]) for layer_name in self.layer_names], dim=1)
        #print('calc act: activations ', activations.shape)

        # calculate neural mean discrepancy
        nmd = activations - nmf

        return features, logits, nmd

class Extract_Features(nn.Module):
    def __init__(self, view_models, layer_names, nmf):
        super().__init__()

        self.nmd = Neural_Mean_Discrepancy(
            view_models=[view_models[0], view_models[1], view_models[2], view_models[3]],
            layer_names=layer_names,
            nmf=nmf)
        for param in self.nmd.parameters():
            param.requires_grad = False
        self.nmd.eval()


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)

        print("Freezing batchnorm layers and stats of base models")
        for m in self.nmd.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
        return self

    def forward(self, x):
        with torch.no_grad():

            nmd, features, logits = self.nmd(x)

        return nmd, features, logits


