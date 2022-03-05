import argparse
import torch
from omegaconf import OmegaConf
from convrnn_classifier import ConvRNNClassifier, ConvRNNConfig
import pandas as pd
from dataset import UtteranceDS, SpecialCollater
from torch.utils.data import DataLoader
from train import TrainConfig, DistributedConfig
from fastprogress.fastprogress import progress_bar
from pathlib import Path

def evaluate(test_csv, ckpt, device):
    device = torch.device(device)
    ckpt = torch.load(ckpt, map_location=device)
    print("Checkpoint keys: ", ckpt.keys())

    cfg = OmegaConf.create(ckpt['cfg_yaml'])
    print("Checkpoint loaded with config ", OmegaConf.to_yaml(cfg))
    model = ConvRNNClassifier(cfg.model_cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_df = pd.read_csv(test_csv)

    testset = UtteranceDS(test_df, cfg.sample_rate)
    testdl = DataLoader(testset, num_workers=cfg.num_workers, shuffle=False,
                            batch_size=cfg.batch_size,
                            pin_memory=False,
                            drop_last=False,
                            collate_fn=SpecialCollater(cfg.seq_len))

    labels = []
    logits = []
    pb = progress_bar(enumerate(testdl), total=len(testdl))
    for i, batch in pb:
        x, xlen, lbls = batch
        x = x.to(device)
        xlen = xlen.to(device)
        lbls = lbls.to(device)

        with torch.no_grad():
            logit = model(x, xlen)
        logits.append(logit)
        labels.append(lbls)
    
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = logits.softmax(dim=-1)
    print(f"Obtained test labels of len {len(labels)} and predicted logits of shape: {logits.shape}.")
    predicted = probs.argmax(dim=-1)
    acc = (predicted == labels).sum()/labels.numel()
    print(f"Accuracy: {acc:4.3f}")

def strip_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    del ckpt['optim_state_dict']
    del ckpt['scheduler_state_dict']
    del ckpt['scaler_state_dict']
    p = Path(ckpt_path)
    po = p.parent/f"{p.stem}-slip.pt"
    torch.save(ckpt, po)
    print("Saved stripped checkpoint to ", str(po))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', required=True, type=str)
    parser.add_argument('--ckpt', required=True, type=str)
    parser.add_argument('--device', default='cuda', type=str)
    a = parser.parse_args()
    evaluate(a.test_csv, a.ckpt, a.device)
    print("Stripping checkpoint.")
    strip_checkpoint(a.ckpt)