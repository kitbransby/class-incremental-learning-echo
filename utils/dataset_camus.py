from torch.utils.data import Dataset
import glob
import numpy as np
import torch
from torchvision.transforms import v2

class CAMUS(Dataset):
    def __init__(self, data_root, transforms, subset, config):

        self.avc_classes = config['AVC_CLASSES']
        self.class_idx_dict = self.class_to_idx()

        self.transforms = transforms
        self.data_root = data_root + subset
        self.norm = config['NORM']
        self.rtn_id = config['RTN_ID']
        self.frame_path_list = glob.glob(data_root + subset + '/*.npy')
        self.mu = 0.19662
        self.std = 0.22815

        print('{} set has {} images'.format(subset, len(self.frame_path_list)))

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
            frame = (frame - self.mu) / self.std
        frame = torch.from_numpy(frame).float()
        label = torch.tensor(label).long()

        if self.shuffle_patches:
            frame = self.shuffle_transforms(frame.unsqueeze(0)).squeeze(0)

        if self.rtn_id:
            return frame, label, id_
        else:
            return frame, label
