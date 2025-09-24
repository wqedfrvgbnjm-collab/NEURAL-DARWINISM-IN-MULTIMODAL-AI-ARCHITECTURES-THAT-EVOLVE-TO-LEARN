import os
import argparse
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np

from evohm.data import get_data_loaders
from evohm.model import EvolutionaryNeuralArchitecture


def parse_args():
    p = argparse.ArgumentParser(description="Eval Evo-Hateful-Memes model")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = get_data_loaders(
        data_root=cfg.data.root,
        train_file=cfg.data.train_file,
        val_file=cfg.data.val_file,
        image_dir=cfg.data.image_dir,
        batch_size=cfg.data.batch_size,
        seq_len=cfg.data.seq_len,
        text_model_name=cfg.model.text_model_name,
        image_model_name=cfg.model.image_model_name,
        use_oversampling_for_train=False,
        num_workers=cfg.data.num_workers,
    )
    model = EvolutionaryNeuralArchitecture(dim=cfg.model.dim, text_model_name=cfg.model.text_model_name, image_model_name=cfg.model.image_model_name).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_data, targets, _ in val_loader:
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            labels = targets['classification'].to(device)
            logits = model(batch_data)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs > 0.5).astype(int)
    print({
        "auc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else 0.0,
        "f1": float(f1_score(labels, preds)) if len(set(labels)) > 1 else 0.0,
        "precision": float(precision_score(labels, preds)) if len(set(labels)) > 1 else 0.0,
        "recall": float(recall_score(labels, preds)) if len(set(labels)) > 1 else 0.0,
    })


if __name__ == "__main__":
    main()
