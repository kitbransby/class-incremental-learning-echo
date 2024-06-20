import argparse
import os
import yaml
import pickle
import numpy as np
import random
from tqdm import tqdm

import torch
from torchinfo import summary

from models.load_model import load_model
from utils.load_dataset import load_dataset
from utils.data_utils import class_maps
from models.neural_mean_discrepancy import Neural_Mean_Discrepancy


def main(config):

    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('connected to device: {}'.format(device))

    wase_model, camus_model, mstr_model, stg_model = load_model(config, device)

    # load view datasets
    camus_config = 'resnet_mcdo_camus_2_class'
    with open('config/' + camus_config + '.yaml') as f:
        camus_config = yaml.load(f, yaml.FullLoader)
    (_, _, _), (camus_train_dataset, _, _) = load_dataset(camus_config)

    mstr_config = 'resnet_mcdo_mstr_11_class'
    with open('config/' + mstr_config + '.yaml') as f:
        mstr_config = yaml.load(f, yaml.FullLoader)
    (_, _, _), (mstr_train_dataset, _, _) = load_dataset(mstr_config)

    wase_config = 'resnet_mcdo_wase_8_class'
    with open('config/' + wase_config + '.yaml') as f:
        wase_config = yaml.load(f, yaml.FullLoader)
    (_, _, _), (wase_train_dataset, _, _) = load_dataset(wase_config)

    stg_config = 'resnet_mcdo_stg_4_class'
    with open('config/' + stg_config + '.yaml') as f:
        stg_config = yaml.load(f, yaml.FullLoader)
    (_, _, _), (stg_train_dataset, _, _) = load_dataset(stg_config)

    models = {'camus': camus_model, 'mstr': mstr_model, 'wase': wase_model, 'stg': stg_model}
    train_datasets = {'camus': camus_train_dataset, 'mstr': mstr_train_dataset, 'wase': wase_train_dataset,
                    'stg': stg_train_dataset}

    layer_names = ['conv1',
                   'layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2',
                   'layer2.0.conv1', 'layer2.0.conv2', 'layer2.1.conv1', 'layer2.1.conv2',
                   'layer3.0.conv1', 'layer3.0.conv2', 'layer3.1.conv1', 'layer3.1.conv2',
                   'layer4.0.conv1', 'layer4.0.conv2', 'layer4.1.conv1', 'layer4.1.conv2', ]

    for model_name, model in models.items():
        print('model/dataset: ' + model_name)
        train_dataset = train_datasets[model_name]
        nmd = Neural_Mean_Discrepancy(model, layer_names, device)
        nmd.fit_in_distribution_dataset(train_dataset)
        features = nmd.id_activations_per_sample.cpu().numpy()
        np.save('results/extracted_features/{}_nmf.npy'.format(model_name), features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_ROOT', type=str)
    parser.add_argument('--CONFIG', type=str)
    config = parser.parse_args()
    cmd_config = vars(config)

    # load model and training configs
    with open('config/' + cmd_config['CONFIG'] + '.yaml') as f:
        yaml_config = yaml.load(f, yaml.FullLoader)

    config = yaml_config
    config.update(cmd_config)  # command line args overide yaml

    print('config: ', config)

    main(config)