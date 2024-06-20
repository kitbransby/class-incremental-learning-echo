from torch.utils.data import Dataset
import numpy as np
import torch
import pickle

class Score_Fusion(Dataset):
    def __init__(self, data_root, transforms, subset, config):

        self.avc_classes = config['AVC_CLASSES']
        self.transforms = transforms
        self.data_root = data_root + subset
        self.use_nmd = config['USE_NMD']

        if self.use_nmd:
            self.nmd = np.concatenate([np.load(data_root + 'WASE_feat/' + subset + '/nmd.npy'),
                                 np.load(data_root + 'CAMUS_feat/' + subset + '/nmd.npy'),
                                 np.load(data_root + 'MSTR_feat/' + subset + '/nmd.npy'),
                                 np.load(data_root + 'STG_feat/' + subset + '/nmd.npy')], axis=0)
        self.logits = np.concatenate([np.load(data_root + 'WASE_feat/' + subset + '/logits.npy'),
                             np.load(data_root + 'CAMUS_feat/' + subset + '/logits.npy'),
                             np.load(data_root + 'MSTR_feat/' + subset + '/logits.npy'),
                             np.load(data_root + 'STG_feat/' + subset + '/logits.npy')], axis=0)
        self.features = np.concatenate([np.load(data_root + 'WASE_feat/' + subset + '/features.npy'),
                             np.load(data_root + 'CAMUS_feat/' + subset + '/features.npy'),
                             np.load(data_root + 'MSTR_feat/' + subset + '/features.npy'),
                             np.load(data_root + 'STG_feat/' + subset + '/features.npy')], axis=0)
        self.labels = np.concatenate([np.load(data_root + 'WASE_feat/' + subset + '/labels.npy'),
                             np.load(data_root + 'CAMUS_feat/' + subset + '/labels.npy'),
                             np.load(data_root + 'MSTR_feat/' + subset + '/labels.npy'),
                             np.load(data_root + 'STG_feat/' + subset + '/labels.npy')], axis=0)

        print('{} set has {} images'.format(subset, len(self.labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        if self.use_nmd:
            nmd = torch.from_numpy(self.nmd[index]).float()
        logits = torch.from_numpy(self.logits[index]).float()
        features = torch.from_numpy(self.features[index]).float()
        label = torch.tensor(self.labels[index]).long()

        if self.weighted:
            return (nmd, features, logits), label
        else:
            return (features, logits), label

class Score_Fusion_Single_Dataset(Dataset):
    def __init__(self, data_root, transforms, subset, config):

        self.avc_classes = config['AVC_CLASSES']
        self.transforms = transforms
        self.data_root = data_root + subset
        print(config)
        self.weighted = config['WEIGHTED']

        if self.weighted:
            self.nmd = np.load(data_root + '/' + subset + '/nmd.npy')
        self.logits = np.load(data_root + '/' + subset + '/logits.npy')
        self.features = np.load(data_root + '/' + subset + '/features.npy')
        self.labels = np.load(data_root + '/' + subset + '/labels.npy')

        print('{} set has {} images'.format(subset, len(self.labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        if self.weighted:
            nmd = torch.from_numpy(self.nmd[index]).float()
        logits = torch.from_numpy(self.logits[index]).float()
        features = torch.from_numpy(self.features[index]).float()
        label = torch.tensor(self.labels[index]).long()

        if self.weighted:
            return (nmd, features, logits), label
        else:
            return (features, logits), label

class Score_Fusion_Aug(Dataset):
    def __init__(self, data_root, transforms, subset, config):

        self.avc_classes = config['AVC_CLASSES']
        self.transforms = transforms
        self.data_root = data_root + subset
        self.weighted = config['WEIGHTED']

        if self.weighted:
            with open(data_root + '/' + subset + '/nmd.pkl', 'rb') as handle:
                self.nmd = pickle.load(handle)
            self.nmd = np.concatenate(self.nmd, axis=0)
        with open(data_root + '/' + subset + '/labels.pkl', 'rb') as handle:
            self.labels = pickle.load(handle)
        self.labels = np.concatenate(self.labels, axis=0)
        with open(data_root + '/' + subset + '/logits.pkl', 'rb') as handle:
            self.logits = pickle.load(handle)
        self.logits = np.concatenate([np.concatenate(logit_list, axis=1) for logit_list in self.logits], axis=0)
        with open(data_root + '/' + subset + '/features.pkl', 'rb') as handle:
            self.features = pickle.load(handle)
        self.features = np.concatenate([np.stack(feature_list, axis=1) for feature_list in self.features], axis=0)

        print('{} set has {} images'.format(subset, len(self.labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        if self.weighted:
            nmd = torch.from_numpy(self.nmd[index]).float()
        logits = torch.from_numpy(self.logits[index]).float()
        features = torch.from_numpy(self.features[index]).float()
        label = torch.tensor(self.labels[index]).long()

        if self.weighted:
            return (nmd, features, logits), label
        else:
            return (features, logits), label