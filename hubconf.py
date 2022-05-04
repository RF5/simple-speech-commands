dependencies = ['torch', 'torchaudio', 'numpy', 'omegaconf']

from dis import pretty_flags
from msilib.schema import ProgId
from random import paretovariate
from convrnn_classifier import ConvRNNClassifier, ConvRNNConfig
from train import TrainConfig, DistributedConfig
from sc09_resnext import ResNextWrapper
from sc09_resnext import CLASSES as RESNEXT_CLASSES
import torch
from omegaconf import OmegaConf

classes = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 
            'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 
            'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 
            'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

def convgru_classifier(pretrained=True, progress=True, device='cuda'):
    r""" 
    GRU classifier model trained Google Speech Commands dataset.
    Model takes in 16kHz raw waveform with samples in the range of [-1, 1].
    Input to model is waveform of shape (bs, T), and output is logits of shape (bs, len(model.classes))
    `model.classes` provides the class mapping for each logit. 

    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        device (str): device to load model onto ('cuda' or 'cpu' are common choices)
    """
    ckpt = torch.hub.load_state_dict_from_url("https://github.com/RF5/simple-speech-commands/releases/download/v0.5/ckpt_00020000-slim.pt", 
                                            progress=progress, map_location=device)
    
    cfg = OmegaConf.create(ckpt['cfg_yaml'])
    model = ConvRNNClassifier(cfg.model_cfg)
    if pretrained:
        model.load_state_dict(ckpt['model_state_dict'])
    model.classes = classes
    model.eval()
    return model

classes_sc09 = ['eight', 'five', 'four', 'nine', 'one', 'seven', 'six', 'three', 'two', 'zero']

def convgru_classifier_sc09(pretrained=True, progress=True, device='cuda', type='best'):
    r""" 
    GRU classifier model trained Google Speech Commands digits dataset (SC09).
    Model takes in 16kHz raw waveform with samples in the range of [-1, 1].
    Input to model is waveform of shape (bs, T), and output is logits of shape (bs, len(model.classes))
    `model.classes` provides the class mapping for each logit. 

    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        device (str): device to load model onto ('cuda' or 'cpu' are common choices)
        type (str): whether to load the "best" checkpoint (lowest validation loss, default), or "last" checkpoint (35k iters).
    """
    if type == 'best':
        ckpt = torch.hub.load_state_dict_from_url("https://github.com/RF5/simple-speech-commands/releases/download/v0.6/ckpt_lowest_validation_00007500-slim.pt", 
                                            progress=progress, map_location=device)
    else:
        ckpt = torch.hub.load_state_dict_from_url("https://github.com/RF5/simple-speech-commands/releases/download/v0.6/ckpt_00035000-slim.pt", 
                                            progress=progress, map_location=device)
    
    cfg = OmegaConf.create(ckpt['cfg_yaml'])
    model = ConvRNNClassifier(cfg.model_cfg)
    if pretrained:
        model.load_state_dict(ckpt['model_state_dict'])
    model.classes = classes_sc09
    model.eval()
    return model


def resnext_classifier_sc09(pretrained=True, progress=True, device='cuda'):
    f"""
    ResNeXT classifier model trained Google Speech Commands digits dataset (SC09).
    Model takes in 16kHz raw waveform with samples in the range of [-1, 1].
    Input to model is waveform of shape (bs, T), and output is logits of shape (bs, len(model.classes))
    `model.classes` provides the class mapping for each logit. 

    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        device (str): device to load model onto ('cuda' or 'cpu' are common choices)
        type (str): whether to load the "best" checkpoint (lowest validation loss, default), or "last" checkpoint (35k iters).
    """
    model = ResNextWrapper()

    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url("https://github.com/RF5/simple-speech-commands/releases/download/v0.7/resnext-export.pt", 
                                            progress=progress, map_location=device)
        model.model.load_state_dict(ckpt)
    model = model.to(device).eval()
    model.classes = RESNEXT_CLASSES
    return model
        