# Minimal demo script (use as reference for a notebook)
import os
import sys
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from evohm.data import get_data_loaders
from evohm.model import EvolutionaryNeuralArchitecture
from evohm.trainer import EvolutionaryTrainer

cfg = OmegaConf.load(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
train_loader, val_loader = get_data_loaders(
    data_root=cfg.data.root,
    train_file=cfg.data.train_file,
    val_file=cfg.data.val_file,
    image_dir=cfg.data.image_dir,
    batch_size=cfg.data.batch_size,
    seq_len=cfg.data.seq_len,
    text_model_name=cfg.model.text_model_name,
    image_model_name=cfg.model.image_model_name,
    use_oversampling_for_train=cfg.data.balance,
    num_workers=cfg.data.num_workers,
)
assert train_loader is not None

model = EvolutionaryNeuralArchitecture(dim=cfg.model.dim, text_model_name=cfg.model.text_model_name, image_model_name=cfg.model.image_model_name)
for init in cfg.model.initial_modules:
    model.add_module(str(init.type), init.reason, dict(init.config))

trainer = EvolutionaryTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    patience=cfg.optim.patience,
    lr=cfg.optim.lr,
    weight_decay=cfg.optim.weight_decay,
    max_lr=cfg.optim.max_lr,
    epochs=1,
    grad_clip=cfg.optim.grad_clip,
    amp=cfg.train.amp,
    evolution_cfg={
        "longevity_threshold": cfg.evolution.longevity_threshold,
        "cooldown_epochs": cfg.evolution.cooldown_epochs,
        "dim": cfg.model.dim,
    },
    output_dir=cfg.output_dir,
)
_ = trainer.train()
