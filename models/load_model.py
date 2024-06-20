import torch
import os
import yaml
import numpy as np

from models.resnet import ResNet18
from models.resnet_ft_expand import ResNet18 as ResNet18_15Class

from models.max_logit import Max_Logit
from models.msp import MSP
from models.confidence_based_routing import Confidence_Based_Routing

from models.extract_features import Extract_Features

from models.score_fusion_model import Score_Fusion
from models.score_fusion_model_nmd import Score_Fusion_NMD
from models.score_fusion_model_attn import Score_Fusion_Attn

def load_model(config, device):
    if config['MODEL'] == 'resnet18':
        model = ResNet18(config['INP_DIM'], len(config['AVC_CLASSES'])).to(device)
    elif config['MODEL'] == 'resnet18_15class':
        model = ResNet18_15Class(config['INP_DIM']).to(device)
    elif config['MODEL'] == 'score_fusion':
        model = Score_Fusion().to(device)
    elif config['MODEL'] ==  'score_fusion_attn':
        model = Score_Fusion_Attn().to(device)
    elif config['MODEL'] == 'score_fusion_nmd':
        model = Score_Fusion_NMD().to(device)
    elif config['MODEL'] == 'pretrained_avcs':
        model = load_pretrained_avc(device)
    elif config['MODEL'] == 'extract_features':
        layer_names = [ 'conv1',
            'layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2',
            'layer2.0.conv1', 'layer2.0.conv2', 'layer2.1.conv1', 'layer2.1.conv2',
            'layer3.0.conv1', 'layer3.0.conv2', 'layer3.1.conv1', 'layer3.1.conv2',
            'layer4.0.conv1', 'layer4.0.conv2', 'layer4.1.conv1', 'layer4.1.conv2',]
        nmf = torch.stack([torch.from_numpy(np.load('results/extracted_features/wase_nmf.npy')),
                           torch.from_numpy(np.load('results/extracted_features/camus_nmf.npy')),
                           torch.from_numpy(np.load('results/extracted_features/mstr_nmf.npy')),
                           torch.from_numpy( np.load('results/extracted_features/stg_nmf.npy'))], dim=0).to(device)
        pretrained_avc_models = load_pretrained_avc(device)
        model = Extract_Features(view_models=pretrained_avc_models, layer_names=layer_names, nmf=nmf).to(device)
    elif config['MODEL'] == 'max_logit':
        pretrained_avc_models = load_pretrained_avc(device)
        model = Max_Logit(view_models=pretrained_avc_models).to(device)
    elif config['MODEL'] == 'msp':
        pretrained_avc_models = load_pretrained_avc(device)
        model = MSP(view_models=pretrained_avc_models).to(device)
    elif config['MODEL'] == 'confidence_based_routing':
        pretrained_avc_models = load_pretrained_avc(device)
        model = Confidence_Based_Routing(view_models=pretrained_avc_models, mode='softmax').to(device)
    else:
        print('Warning: No model selected')
    return model

def load_pretrained_avc(device):
    # load view classifiers and datasets
    camus_config = 'camus_2_class'
    with open('config/' + camus_config + '.yaml') as f:
        camus_config = yaml.load(f, yaml.FullLoader)
    camus_config['RUN_ID'] = '<ADD_CAMUS_WEIGHTS_RUN_ID_HERE'
    camus_model = load_model(camus_config, device)
    camus_model.load_state_dict(
        torch.load(os.path.join('results', camus_config['RUN_ID'], 'best_acc.pt')))
    camus_model.eval()

    mstr_config = 'mstr_11_class'
    with open('config/' + mstr_config + '.yaml') as f:
        mstr_config = yaml.load(f, yaml.FullLoader)
    mstr_config['RUN_ID'] = 'ADD_MSTR_WEIGHTS_RUN_ID_HERE'
    mstr_model = load_model(mstr_config, device)
    mstr_model.load_state_dict(
        torch.load(os.path.join('results', mstr_config['RUN_ID'], 'best_acc.pt')))
    mstr_model.eval()

    wase_config = 'wase_8_class'
    with open('/config/' + wase_config + '.yaml') as f:
        wase_config = yaml.load(f, yaml.FullLoader)
    wase_config['RUN_ID'] = 'ADD_WASE_WEIGHTS_RUN_ID_HERE'
    wase_model = load_model(wase_config, device)
    wase_model.load_state_dict(
        torch.load(os.path.join('results', wase_config['RUN_ID'], 'best_acc.pt')))
    wase_model.eval()

    stg_config = 'stg_4_class'
    with open('/config/' + stg_config + '.yaml') as f:
        stg_config = yaml.load(f, yaml.FullLoader)
    stg_config['RUN_ID'] = 'ADD_STG_WEIGHTS_RUN_ID_HERE'
    stg_model = load_model(stg_config, device)
    stg_model.load_state_dict(
        torch.load(os.path.join('results', stg_config['RUN_ID'], 'best_acc.pt')))
    stg_model.eval()

    return wase_model, camus_model, mstr_model, stg_model