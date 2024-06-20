import argparse
import os
import yaml
import datetime
import time
import pickle
import sys
import numpy as np
import random
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from models.load_model import load_model

from utils.train_utils import plot
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

    AVC_CLASSES_all = ['contrast-A2C', 'contrast-A3C', 'contrast-A4C', 'contrast-PLAX',
              'noncontrast-A2C', 'noncontrast-A3C', 'noncontrast-A4C', 'noncontrast-A5C',
              'noncontrast-PLAX','noncontrast-PLAX-AV', 'noncontrast-PSAX-AV', 'noncontrast-PSAX-PM',
              'noncontrast-RV', 'noncontrast-SC','noncontrast-SC-IVC']
    AVC_CLASSES_train_idx = [i for i, a in enumerate(AVC_CLASSES_all) if a in config['AVC_CLASSES']]
    AVC_CLASSES_freeze_idx = [i for i, a in enumerate(AVC_CLASSES_all) if a not in config['AVC_CLASSES']]
    print('Freezing these classes idx: ', AVC_CLASSES_freeze_idx)
    model = load_model(config, device)

    if config['WEIGHTS'] is not None:
        weight_path = os.path.join( "results", config['WEIGHTS'], 'best_acc.pt')
        pretrained_weights = torch.load(weight_path)
        model.load_state_dict(pretrained_weights, strict=True)
        print('Load weights: ', weight_path)

    for name, param in model.named_parameters():
        if name.startswith('linear') and int(name.split('.')[1]) in AVC_CLASSES_freeze_idx:
            print('freezing layer', name)
            param.requires_grad = False

    summary(model, input_size=(config['BATCH_SIZE'], config['INP_DIM'], config['RESOLUTION'], config['RESOLUTION']))

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), config['LR'])
    max_epochs = config['EPOCHS']

    save_folder = os.path.join( "results", config['RUN_ID'])

    best_accuracy = 0
    train_loss_all, val_loss_all = [], []
    train_acc_all, val_acc_all = [], []

    print('Starting Training...')
    for epoch in range(max_epochs):
        start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        correct_pred = 0
        incorrect_pred = 0
        for X, Y, id_ in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()

            Y_hat = model(X)

            Y_hat = Y_hat[:, AVC_CLASSES_train_idx]

            loss = loss_function(Y_hat, Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            correct_pred += (Y_hat.argmax(dim=1) == Y).sum()
            incorrect_pred += (Y_hat.argmax(dim=1) != Y).sum()
            if config['VERBOSE']:
                if step % config['TRAIN_PRINT'] == 0:
                    print(f"{step}/{len(train_dataset) // train_loader.batch_size}, " f"train_loss: {loss.item():.5f}")
            step += 1
            #break
        epoch_loss /= step
        train_loss_all.append(epoch_loss)
        accuracy = correct_pred / (correct_pred + incorrect_pred)
        train_acc_all.append(accuracy.cpu().numpy())
        print(f"Train epoch: {epoch + 1} avg loss: {epoch_loss:.4f}, avg acc: {accuracy:.2f}" )


        model.eval()
        with torch.no_grad():
            epoch_loss = 0
            step = 0
            correct_pred = 0
            incorrect_pred = 0
            for X, Y, id_ in val_loader:
                X, Y = X.to(device),Y.to(device)
                Y_hat = model(X)
                Y_hat = Y_hat[:, AVC_CLASSES_train_idx]
                loss = loss_function(Y_hat, Y)
                epoch_loss += loss.item()
                correct_pred += (Y_hat.argmax(dim=1) == Y).sum()
                incorrect_pred += (Y_hat.argmax(dim=1) != Y).sum()
                if config['VERBOSE']:
                    if step % config['VAL_PRINT'] == 0:
                        print(f"{step}/{len(val_dataset) // val_loader.batch_size}, " f"val_loss: {loss.item():.5f}")
                step += 1
                #break
            epoch_loss /= step
            val_loss_all.append(epoch_loss)
            accuracy = correct_pred / (correct_pred + incorrect_pred)
            val_acc_all.append(accuracy.cpu().numpy())
            print(f"Val epoch: {epoch + 1} avg loss: {epoch_loss:.4f}, avg acc: {accuracy:.2f}" )
            end = time.time()
            epoch_time = end - start
            print('Epoch time: {:.2f}s'.format(epoch_time))
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), os.path.join(save_folder, 'best_acc.pt'))
                print("saved model new best acc")

        #print('Training done, saving logs to {}'.format(save_folder))
        with open(save_folder+'/losses.pkl', 'wb') as f:
            pickle.dump([train_loss_all, val_loss_all,train_acc_all, val_acc_all], f)

        plot(train_loss_all, val_loss_all,train_acc_all, val_acc_all, save_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_ROOT', type=str)
    parser.add_argument('--CONFIG', type=str)
    parser.add_argument('--RUN_ID', type=str)
    config = parser.parse_args()
    cmd_config = vars(config)

    # load model and training configs
    with open('config/' + cmd_config['CONFIG'] + '.yaml') as f:
        yaml_config = yaml.load(f, yaml.FullLoader)

    config = yaml_config
    config.update(cmd_config)  # command line args overide yaml

    print('config: ', config)

    main(config)
