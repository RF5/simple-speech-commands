dependencies = ['torch', 'torchaudio', 'numpy', 'omegaconf']

from convrnn_classifier import ConvRNNClassifier, ConvRNNConfig
from train import TrainConfig, DistributedConfig
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

def convgru_classifier_sc09(pretrained=True, progress=True, device='cuda'):
    r""" 
    GRU classifier model trained Google Speech Commands digits dataset (SC09).
    Model takes in 16kHz raw waveform with samples in the range of [-1, 1].
    Input to model is waveform of shape (bs, T), and output is logits of shape (bs, len(model.classes))
    `model.classes` provides the class mapping for each logit. 

    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        device (str): device to load model onto ('cuda' or 'cpu' are common choices)
    """
    ckpt = torch.hub.load_state_dict_from_url("https://github.com/RF5/simple-speech-commands/releases/download/v0.6/ckpt_00047500-slim.pt", 
                                            progress=progress, map_location=device)
    
    cfg = OmegaConf.create(ckpt['cfg_yaml'])
    model = ConvRNNClassifier(cfg.model_cfg)
    if pretrained:
        model.load_state_dict(ckpt['model_state_dict'])
    model.classes = classes_sc09
    model.eval()
    return model
