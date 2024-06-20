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

    model = load_model(config, device)

    summary(model, input_size=(config['BATCH_SIZE'], config['INP_DIM'], config['RESOLUTION'], config['RESOLUTION']))

    # load view datasets
    camus_config = 'resnet_mcdo_camus_2_class'
    with open('config/' + camus_config + '.yaml') as f:
        camus_config = yaml.load(f, yaml.FullLoader)

    mstr_config = 'resnet_mcdo_mstr_11_class'
    with open('config/' + mstr_config + '.yaml') as f:
        mstr_config = yaml.load(f, yaml.FullLoader)

    wase_config = 'resnet_mcdo_wase_8_class'
    with open('config/' + wase_config + '.yaml') as f:
        wase_config = yaml.load(f, yaml.FullLoader)

    stg_config = 'resnet_mcdo_stg_4_class'
    with open('config/' + stg_config + '.yaml') as f:
        stg_config = yaml.load(f, yaml.FullLoader)

    mahi_config = 'resnet_mcdo_mahi_15_class'
    with open('config/' + mahi_config + '.yaml') as f:
        mahi_config = yaml.load(f, yaml.FullLoader)

    uoc_config = 'resnet_mcdo_uoc_15_class'
    with open('config/' + uoc_config + '.yaml') as f:
        uoc_config = yaml.load(f, yaml.FullLoader)


    for dataset_id, dataset_config in enumerate(
            [wase_config, camus_config, mstr_config, stg_config, mahi_config, uoc_config]):
        print('Evaluating {}'.format(dataset_config['DATASET']))
        class_map = class_maps(dataset_id)

        dataset_config['TRAIN_TRANSFORMS'] = False

        (_, _, _), (_, val_dataset, test_dataset) = load_dataset(dataset_config)

        for dataset, set_str in [[val_dataset, 'val'], [test_dataset, 'test']]:

            labels_npy = np.zeros(len(dataset), dtype=np.int32)
            nmd_npy = np.zeros((len(dataset), 15616), dtype=np.float32)
            features_npy = np.zeros((len(dataset), 4, 512), dtype=np.float32)
            logits_npy = np.zeros((len(dataset), 25), dtype=np.float32)

            for i in tqdm(range(len(dataset))):
                X, Y = dataset[i]
                X = X.unsqueeze(0).to(device)

                Y = class_map[int(Y.numpy())]

                nmd, features, logits = model(X)

                nmd_npy[i, :] = nmd[0, :].cpu().numpy()
                features_npy[i, :, :] = torch.cat(features, dim=0).cpu().numpy()
                logits_npy[i, :] = torch.cat(logits, dim=1).squeeze(0).cpu().numpy()
                labels_npy[i] = Y

            np.save(config['DATA_ROOT'] + '{}_feat/{}/labels.npy'.format(
                dataset_config['DATASET'].upper(), set_str), labels_npy)
            np.save(config['DATA_ROOT'] + '{}_feat/{}/nmd.npy'.format(
                dataset_config['DATASET'].upper(), set_str), nmd_npy)
            np.save(config['DATA_ROOT'] + '{}_feat/{}/features.npy'.format(
                dataset_config['DATASET'].upper(), set_str), features_npy)
            np.save(config['DATA_ROOT'] + '{}_feat/{}/logits.npy'.format(
                dataset_config['DATASET'].upper(), set_str), logits_npy)


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