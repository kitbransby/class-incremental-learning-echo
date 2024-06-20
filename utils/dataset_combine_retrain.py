from torch.utils.data import Dataset
import numpy as np
import torch

from utils.dataset_stg import STG
from utils.dataset_wase import WASE
from utils.dataset_camus import CAMUS
from utils.dataset_mstr import MSTR


class Combine_Retrain(Dataset):
    def __init__(self, data_root, transforms, subset, config):

        self.avc_classes = config['AVC_CLASSES']
        self.class_idx_dict = self.class_to_idx()

        self.transforms = transforms
        self.data_root = data_root + subset
        self.norm = config['NORM']
        self.rtn_id = config['RTN_ID']
        self.rtn_dataset_id = config['RTN_DATASET_ID']

        config['AVC_CLASSES'] = ['noncontrast-A2C', 'noncontrast-A4C']
        self.CAMUS = CAMUS(data_root + 'CAMUS/', transforms, subset, config)
        config['AVC_CLASSES'] = ['noncontrast-A2C', 'noncontrast-A3C', 'noncontrast-A4C', 'noncontrast-A5C',
              'noncontrast-PLAX','noncontrast-PLAX-AV', 'noncontrast-PSAX-AV', 'noncontrast-PSAX-PM',
              'noncontrast-RV','noncontrast-SC','noncontrast-SC-IVC']
        self.MSTR = MSTR(data_root + 'MSTR/', transforms, subset, config)
        config['AVC_CLASSES'] = ['noncontrast-A2C', 'noncontrast-A3C', 'noncontrast-A4C', 'noncontrast-A5C',
              'noncontrast-PLAX','noncontrast-PLAX-AV', 'noncontrast-PSAX-AV', 'noncontrast-PSAX-PM']
        self.WASE = WASE(data_root + 'WASE_11_Class/', transforms, subset, config)
        config['AVC_CLASSES'] = ['contrast-A2C', 'contrast-A3C', 'contrast-A4C', 'contrast-PLAX']
        self.STG = STG(data_root + 'STG/', transforms, subset, config)

        config['AVC_CLASSES'] = self.avc_classes
        #print(config['AVC_CLASSES'])

        self.camus_frames = self.CAMUS.frame_path_list
        self.mstr_frames = self.MSTR.frame_path_list
        self.wase_frames = self.WASE.frame_path_list
        self.stg_frames = self.STG.frame_path_list

        print('{} set has {} stg images \n'.format(subset, len(self.stg_frames)))
        print('{} set has {} wase images \n'.format(subset, len(self.wase_frames)))
        print('{} set has {} mstr images \n'.format(subset, len(self.mstr_frames)))
        print('{} set has {} camus images \n'.format(subset, len(self.camus_frames)))

        self.frame_path_list = self.stg_frames + self.wase_frames + self.mstr_frames + self.camus_frames
        self.dataset_ids = ([0 for i in self.stg_frames] +
                            [1 for i in self.wase_frames] +
                            [2 for i in self.mstr_frames] +
                            [3 for i in self.camus_frames])
        print('{} set has {} total images \n'.format(subset, len(self.frame_path_list)))

    def __len__(self):
        return len(self.frame_path_list)

    def class_to_idx(self):
        class_idx_dict = {}
        for i, cls in enumerate(self.avc_classes):
            class_idx_dict[cls] = i
        return class_idx_dict

    def __getitem__(self, index):

        # load frame, label, id, noise
        file_name = self.frame_path_list[index]
        dataset_id = self.dataset_ids[index]
        frame = np.load(file_name).astype(np.uint8)
        label = self.class_idx_dict[file_name.split('_')[-1].split('.')[0]]
        id_ = file_name.split('/')[-1]

        # augmentation
        if self.transforms is not None:
            augmented = self.transforms(image=frame)
            frame = augmented['image']

        frame = frame.astype(np.float32)

        # format
        frame = np.expand_dims(frame, axis=0)
        frame = np.clip(frame, 0, 255) / 255
        if self.norm == 'z':
            if dataset_id == 0:
                mu, std = self.STG.mu, self.STG.std
            elif dataset_id == 1:
                mu, std = self.WASE.mu, self.WASE.std
            elif dataset_id == 2:
                mu, std = self.MSTR.mu, self.MSTR.std
            elif dataset_id == 3:
                mu, std = self.CAMUS.mu, self.CAMUS.std
            else:
                print('ERROR: Unknown dataset')
            frame = (frame - mu) / std
        frame = torch.from_numpy(frame).float()
        label = torch.tensor(label).long()

        if self.rtn_id:
            if self.rtn_dataset_id:
                return frame, label, id_, dataset_id
            else:
                return frame, label, id_
        else:
            return frame, label
