import argparse
import os
import yaml
import numpy as np
import time
import pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F

from utils.load_dataset import load_dataset
from models.load_model import load_model
from utils.data_utils import class_maps

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score


def main(config):

    all_classes = ['contrast-A2C', 'contrast-A3C', 'contrast-A4C', 'contrast-PLAX',
              'noncontrast-A2C', 'noncontrast-A3C', 'noncontrast-A4C', 'noncontrast-A5C',
              'noncontrast-PLAX','noncontrast-PLAX-AV', 'noncontrast-PSAX-AV', 'noncontrast-PSAX-PM',
              'noncontrast-RV','noncontrast-SC','noncontrast-SC-IVC']

    camus_config = 'resnet_mcdo_camus_2_class'
    with open('/config/' + camus_config + '.yaml') as f:
        camus_config = yaml.load(f, yaml.FullLoader)
    camus_config['DATASET'] = 'CAMUS_feat'
    camus_config['WEIGHTED'] = config['WEIGHTED']
    mstr_config = 'resnet_mcdo_mstr_11_class'
    with open('/config/' + mstr_config + '.yaml') as f:
        mstr_config = yaml.load(f, yaml.FullLoader)
    mstr_config['DATASET'] = 'MSTR_feat'
    mstr_config['WEIGHTED'] = config['WEIGHTED']
    wase_config = 'resnet_mcdo_wase_8_class'
    with open('/config/' + wase_config + '.yaml') as f:
        wase_config = yaml.load(f, yaml.FullLoader)
    wase_config['DATASET'] = 'WASE_feat'
    wase_config['WEIGHTED'] = config['WEIGHTED']
    stg_config = 'resnet_mcdo_stg_4_class'
    with open('/config/' + stg_config + '.yaml') as f:
        stg_config = yaml.load(f, yaml.FullLoader)
    stg_config['DATASET'] = 'STG_feat'
    stg_config['WEIGHTED'] = config['WEIGHTED']
    mahi_config = camus_config.copy()
    mahi_config['DATASET'] = 'MAHI_feat'
    mahi_config['AVC_CLASSES'] = all_classes
    uoc_config = mahi_config.copy()
    uoc_config['DATASET'] = 'UOC_feat'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('connected to device: {}'.format(device))

    model = load_model(config, device)
    model.load_state_dict(torch.load(os.path.join('', 'results', config['RUN_ID'], 'best_acc.pt')))

    Y_pred_testset1 = []
    Y_true_testset1 = []
    Y_pred_testset2 = []
    Y_true_testset2 = []

    if config['PRED_CLASS_MAP_ID'] is not None:
        pred_class_map = class_maps(config['PRED_CLASS_MAP_ID'])

    config['AVC_CLASSES'] = all_classes

    for dataset_id, dataset_config in enumerate([wase_config, camus_config, mstr_config, stg_config, mahi_config, uoc_config]):
        print('Evaluating {}'.format(dataset_config['DATASET']))
        class_map = class_maps(dataset_id)

        (_, _, _), (_, _, test_dataset) = load_dataset(dataset_config)

        save_folder = os.path.join('', "results", config['RUN_ID'], 'evaluation_' + dataset_config['DATASET'])
        try:
            os.mkdir(save_folder)
        except Exception as e:
            print('Warning: ', e)

        Y_pred_all = np.zeros(len(test_dataset), dtype=np.int32)
        Y_true_all = np.zeros(len(test_dataset), dtype=np.int32)
        intermediate_all = []
        avg_time = []

        model.eval()
        with torch.no_grad():
            for i in tqdm(range(len(test_dataset))):

                X, Y = test_dataset[i]

                if len(X) == 3:
                    nmd, features, logits = X
                    nmd, features, logits = nmd.unsqueeze(0).to(device), features.unsqueeze(0).to(device), logits.unsqueeze(0).to(device)
                    start = time.time()
                    output = model(nmd, features, logits)
                    end = time.time()
                elif len(X) == 2:
                    features, logits = X
                    features, logits = features.unsqueeze(0).to(device), logits.unsqueeze(0).to(device)
                    start = time.time()
                    output = model(features, logits)
                    end = time.time()
                else:
                    print('Error.. length of X: ', len(X))
                avg_time.append(end - start)

                # some of the models output a single integer rather than a logit/softmax as the final view prediction.
                if isinstance(output, int):
                    Y_pred = output

                else:
                    Y_logits, intermediate = output
                    Y_softmax = F.softmax(Y_logits, dim=1).cpu().numpy()
                    Y_pred = np.argmax(Y_softmax, axis=1)[0]
                    # the original view classifers do not predict 15 classes.
                    if Y_softmax.shape[1] < 15:
                        Y_pred = pred_class_map[int(Y_pred)]

                    intermediate_all.append(intermediate + [Y])

                Y_pred_all[i] = Y_pred
                Y_true_all[i] = Y

        if dataset_config['DATASET'] in ['WASE_feat', 'CAMUS_feat', 'MSTR_feat', 'STG_feat']:
            Y_pred_testset1.append(Y_pred_all)
            Y_true_testset1.append(Y_true_all)
        if dataset_config['DATASET'] in ['MAHI_feat', 'UOC_feat']:
            Y_pred_testset2.append(Y_pred_all)
            Y_true_testset2.append(Y_true_all)


        clf_report = classification_report(Y_true_all, Y_pred_all, labels=list(range(len(config['AVC_CLASSES']))),
                                           target_names=config['AVC_CLASSES'], digits=4, output_dict=False, zero_division='warn')
        print(clf_report)
        results = {'clf_report': clf_report, 'speed': avg_time}

        print('Saving predictions and scores to {}'.format(save_folder))
        with open(save_folder + '/scores.pkl', 'wb') as f:
            pickle.dump(results, f)
        with open(save_folder+'/predictions.pkl', 'wb') as f:
            pickle.dump([Y_true_all, Y_pred_all], f)
        with open(save_folder + '/logits.pkl', 'wb') as f:
                pickle.dump(intermediate_all, f)


    print('Evaluating all Test set 1 (WASE, CAMUS, MSTR, STG)...')
    Y_pred_testset1_all = np.concatenate(Y_pred_testset1, axis=0)
    Y_true_testset1_all = np.concatenate(Y_true_testset1, axis=0)
    clf_report = classification_report(Y_true_testset1_all, Y_pred_testset1_all, labels=list(range(len(config['AVC_CLASSES']))),
                                       target_names=config['AVC_CLASSES'], digits=4, output_dict=False,
                                       zero_division='warn')
    print(clf_report)
    save_folder = os.path.join('', "results", config['RUN_ID'], 'evaluation_wase_camus_mstr_stg')
    try:
        os.mkdir(save_folder)
    except Exception as e:
        print('Warning: ', e)
    results = {'clf_report': clf_report}
    print('Saving predictions and scores to {}'.format(save_folder))
    with open(save_folder + '/scores.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open(save_folder + '/predictions.pkl', 'wb') as f:
        pickle.dump([Y_pred_testset1_all, Y_true_testset1_all], f)

    print('Evaluating all Test set 2 (MAHI, UOC)...')
    Y_pred_testset2_all = np.concatenate(Y_pred_testset2, axis=0)
    Y_true_testset2_all = np.concatenate(Y_true_testset2, axis=0)
    clf_report = classification_report(Y_true_testset2_all, Y_pred_testset2_all,
                                       labels=list(range(len(config['AVC_CLASSES']))),
                                       target_names=config['AVC_CLASSES'], digits=4, output_dict=False,
                                       zero_division='warn')
    print(clf_report)
    save_folder = os.path.join('', "results", config['RUN_ID'], 'evaluation_mahi_uoc')
    try:
        os.mkdir(save_folder)
    except Exception as e:
        print('Warning: ', e)
    results = {'clf_report': clf_report}
    print('Saving predictions and scores to {}'.format(save_folder))
    with open(save_folder + '/scores.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open(save_folder + '/predictions.pkl', 'wb') as f:
        pickle.dump([Y_pred_testset2_all, Y_true_testset2_all], f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_ROOT', type=str)
    parser.add_argument('--RUN_ID', type=str)
    parser.add_argument('--CONFIG', type=str)
    config = parser.parse_args()
    cmd_config = vars(config)

    # load model and training configs
    with open('/config/' + cmd_config['CONFIG'] + '.yaml') as f:
        yaml_config = yaml.load(f, yaml.FullLoader)

    config = yaml_config
    config.update(cmd_config)  # command line args overide yaml

    print('config: ', config)

    main(config)
