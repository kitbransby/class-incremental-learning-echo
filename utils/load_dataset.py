from utils.dataset_camus import CAMUS
from utils.dataset_mstr import MSTR
from utils.dataset_combine_retrain import Combine_Retrain
from utils.dataset_stg import STG
from utils.dataset_wase import WASE
from utils.dataset_uoc import UoC
from utils.dataset_mahi import MAHI
from utils.dataset_score_fusion import Score_Fusion_Single_Dataset, Score_Fusion, Score_Fusion_Aug

from torch.utils.data import DataLoader
import albumentations as A

import torch
import numpy as np
import random


def load_dataset(config):

    data_root = config['DATA_ROOT']

    if config['TRAIN_TRANSFORMS']:
        train_transforms = A.Compose(
            [A.Rotate(30, p=1, interpolation=2, border_mode=1),
             A.HorizontalFlip(p=0.5),
             A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1)])
    else:
        train_transforms = None

    if config['DATASET'] == 'camus':
        train_dataset = CAMUS(data_root=data_root + 'CAMUS/',
                                      transforms=train_transforms,
                                      subset='train',
                                      config=config)
        val_dataset = CAMUS(data_root=data_root + 'CAMUS/',
                                    transforms=None,
                                    subset='val',
                                    config=config)
        test_dataset = CAMUS(data_root=data_root + 'CAMUS/',
                                     transforms=None,
                                     subset='test',
                                     config=config)
    elif config['DATASET'] == 'mstr':
        train_dataset = MSTR(data_root=data_root + 'MSTR/',
                                      transforms=train_transforms,
                                      subset='train',
                                      config=config)
        val_dataset = MSTR(data_root=data_root + 'MSTR/',
                                    transforms=None,
                                    subset='val',
                                    config=config)
        test_dataset = MSTR(data_root=data_root + 'MSTR/',
                                     transforms=None,
                                     subset='test',
                                     config=config)
    elif config['DATASET'] == 'wase':
        train_dataset = WASE(data_root=data_root + 'WASE_11_Class/',
                                      transforms=train_transforms,
                                      subset='train',
                                      config=config)
        val_dataset = WASE(data_root=data_root + 'WASE_11_Class/',
                                    transforms=None,
                                    subset='val',
                                    config=config)
        test_dataset = WASE(data_root=data_root + 'WASE_11_Class/',
                                     transforms=None,
                                     subset='test',
                                     config=config)
    elif config['DATASET'] == 'stg':
        train_dataset = STG(data_root=data_root + 'STG/',
                                      transforms=train_transforms,
                                      subset='train',
                                      config=config)
        val_dataset = STG(data_root=data_root + 'STG/',
                                    transforms=None,
                                    subset='val',
                                    config=config)
        test_dataset = STG(data_root=data_root + 'STG/',
                                     transforms=None,
                                     subset='test',
                                     config=config)
    elif config['DATASET'] == 'combine_retrain':
        train_dataset = Combine_Retrain(data_root=data_root,
                                      transforms=train_transforms,
                                      subset='train',
                                      config=config)
        val_dataset = Combine_Retrain(data_root=data_root,
                                    transforms=None,
                                    subset='val',
                                    config=config)
        test_dataset = Combine_Retrain(data_root=data_root,
                                     transforms=None,
                                     subset='test',
                                     config=config)
    elif config['DATASET'] == 'mahi':
        train_dataset = MAHI(data_root=data_root + 'MAHI/',
                                      transforms=train_transforms,
                                      subset='train',
                                      config=config)
        val_dataset = MAHI(data_root=data_root + 'MAHI/',
                                    transforms=None,
                                    subset='val',
                                    config=config)
        test_dataset = MAHI(data_root=data_root + 'MAHI/',
                                     transforms=None,
                                     subset='test',
                                     config=config)
    elif config['DATASET'] == 'uoc':
        train_dataset = UoC(data_root=data_root + 'UoC_cardiomyopathy/',
                                      transforms=train_transforms,
                                      subset='train',
                                      config=config)
        val_dataset = UoC(data_root=data_root + 'UoC_cardiomyopathy/',
                                    transforms=None,
                                    subset='val',
                                    config=config)
        test_dataset = UoC(data_root=data_root + 'UoC_cardiomyopathy/',
                                     transforms=None,
                                     subset='test',
                                     config=config)
    elif config['DATASET'] == 'kcvi':
        train_dataset = KCVI(data_root=data_root + 'KCVI/',
                                      transforms=train_transforms,
                                      subset='train',
                                      config=config)
        val_dataset = KCVI(data_root=data_root + 'KCVI/',
                                    transforms=None,
                                    subset='val',
                                    config=config)
        test_dataset = KCVI(data_root=data_root + 'KCVI/',
                                     transforms=None,
                                     subset='test',
                                     config=config)
    elif 'feat' in config['DATASET']:
        train_dataset = Score_Fusion_Single_Dataset(data_root=data_root + 'Score_Fusion/' + config['DATASET'],
                                     transforms=None,
                                     subset='train',
                                     config=config)
        val_dataset = Score_Fusion_Single_Dataset(data_root=data_root + 'Score_Fusion/' + config['DATASET'],
                                   transforms=None,
                                   subset='val',
                                   config=config)
        test_dataset = Score_Fusion_Single_Dataset(data_root=data_root + 'Score_Fusion/' + config['DATASET'],
                                    transforms=None,
                                    subset='test',
                                    config=config)
    elif config['DATASET'] == 'score_fusion':
        train_dataset = Score_Fusion_Aug(data_root=data_root + 'Score_Fusion_Aug',
                                      transforms=None,
                                      subset='train',
                                      config=config)
        val_dataset = Score_Fusion(data_root=data_root + 'Score_Fusion/',
                                    transforms=None,
                                    subset='val',
                                    config=config)
        test_dataset = Score_Fusion(data_root=data_root + 'Score_Fusion/',
                                     transforms=None,
                                     subset='test',
                                     config=config)

    else:
        print('WARNING - No dataset selected..')

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True,
                              num_workers=config['NUM_WORKERS'], pin_memory=True, worker_init_fn=seed_worker,generator=g)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], num_workers=config['NUM_WORKERS'],
                            pin_memory=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], num_workers=config['NUM_WORKERS'],
                             pin_memory=True, worker_init_fn=seed_worker, generator=g)

    return (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset)