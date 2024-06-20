import argparse
import os
import yaml
import pickle
import numpy as np
import random

import torch
from torchinfo import summary

from models.load_model import load_model
from utils.load_dataset import load_dataset


def main(config):

    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset) = load_dataset(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('connected to device: {}'.format(device))

    model = load_model(config, device)

    summary(model, input_size=(config['BATCH_SIZE'], config['INP_DIM'], config['RESOLUTION'], config['RESOLUTION']))

    optimizer = torch.optim.Adam(model.parameters(), config['LR'])
    max_epochs = config['EPOCHS']

    save_folder = os.path.join("results", config['RUN_ID'])

    nmd_all = []
    features_all = []
    logits_all = []
    labels_all = []

    print('Starting Training...')
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()

        for X, Y, id_ in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()

            nmd, features, logits = model(X)

            nmd_all.append(nmd.cpu().numpy())
            features_all.append([f.cpu().numpy() for f in features])
            logits_all.append([l.cpu().numpy() for l in logits])
            labels_all.append(Y.cpu().numpy())

        with open(save_folder+'/nmd.pkl', 'wb') as f:
            pickle.dump(nmd_all, f)
        with open(save_folder+'/features.pkl', 'wb') as f:
            pickle.dump(features_all, f)
        with open(save_folder+'/logits.pkl', 'wb') as f:
            pickle.dump(logits_all, f)
        with open(save_folder+'/labels.pkl', 'wb') as f:
            pickle.dump(labels_all, f)

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
