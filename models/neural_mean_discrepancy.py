import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Neural_Mean_Discrepancy(nn.Module):
    def __init__(self, model, layer_names, device):
        super().__init__()
        self.model = model
        self.model.eval()
        self.device = device
        self.layer_names = layer_names
        self.activation = {}
        # register the hooks using the requested layer names
        self.register_activations()

    def get_activations(self, name):
        # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def register_activations(self):
        # register a forward hook for every requested layer name
        for name, layer in self.model.named_modules():
            if name in self.layer_names:
                layer.register_forward_hook(self.get_activations(name))

    def fit_in_distribution_dataset(self, id_dataset):
        print('Fitting in-distribution dataset..')
        self.id_activations_per_sample, self.id_activations = self.calc_activations(id_dataset)

    def fit_unknown_distribution_dataset(self, ud_dataset):
        print('Fitting unknown distribution dataset..')
        self.ud_activations_per_sample, self.ud_activations = self.calc_activations(ud_dataset)

    def calc_activations(self, dataset):

        # create empty dictionary to store activations for every example
        layer_activations = {key: [] for key in self.layer_names}

        # iterate through dataset
        for (x, y) in tqdm(dataset):

            # pass single input through model
            _ = self.model(x.to(self.device).unsqueeze(0))

            # iterate through all the layers we need activations for
            for layer_name in self.layer_names:
                # get activation map
                activation_map = self.activation[layer_name]

                # take mean over the spatial dims
                channel_activations = activation_map.mean(dim=[0, 2, 3])

                # append to layer activation dictionary
                layer_activations[layer_name].append(channel_activations)

        # stack activations per sample
        layer_activations_per_sample = torch.cat(
            [torch.stack(activations, dim=0) for layer_name, activations in layer_activations.items()],
            dim=1)

        # take mean over all examples, and concat to a single vector
        layer_activations_avgd = torch.cat(
            [torch.stack(activations, dim=0).mean(dim=0) for layer_name, activations in layer_activations.items()],
            dim=0)

        return layer_activations_per_sample, layer_activations_avgd

    def score(self):
        diff = self.id_activations - self.ud_activations
        mae = (self.id_activations - self.ud_activations).abs().mean()
        return diff, mae



