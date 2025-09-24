import os
import argparse
import json
import torch
from omegaconf import OmegaConf

from evohm.data import get_data_loaders
from evohm.model import EvolutionaryNeuralArchitecture
from evohm.trainer import EvolutionaryTrainer
from evohm.modules import ModuleType


def parse_args():
    parser = argparse.ArgumentParser(description="Train Evo-Hateful-Memes model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--override", type=str, nargs="*", default=None, help="Override config as key=value pairs")
    return parser.parse_args()


def load_config(path: str, overrides=None):
    cfg = OmegaConf.load(path)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)

    torch.manual_seed(cfg.get("seed", 42))
    torch.cuda.manual_seed_all(cfg.get("seed", 42))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    if train_loader is None:
        print("No training data found. Check dataset paths in config.")
        return

    model = EvolutionaryNeuralArchitecture(dim=cfg.model.dim, text_model_name=cfg.model.text_model_name, image_model_name=cfg.model.image_model_name)
    for init in cfg.model.initial_modules:
        mtype = str(init.type)
        module = model.add_module(mtype, init.reason, dict(init.config))
    wandb_cfg = dict(cfg.get("wandb", {}))
    trainer = EvolutionaryTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        patience=cfg.optim.patience,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        max_lr=cfg.optim.max_lr,
        epochs=cfg.optim.epochs,
        grad_clip=cfg.optim.grad_clip,
        amp=cfg.train.amp,
        evolution_cfg={
            "longevity_threshold": cfg.evolution.longevity_threshold,
            "cooldown_epochs": cfg.evolution.cooldown_epochs,
            "dim": cfg.model.dim,
        },
        output_dir=cfg.output_dir,
        wandb_cfg=wandb_cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()
