import os
import json
import time
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Optional Weights & Biases
try:
    import wandb  # type: ignore
except Exception:
    wandb = None

from .losses import FocalLoss
from .strategist import EvolutionaryStrategist


class EpochMetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.epoch_loss, self.batch_count = 0.0, 0
        self.all_preds, self.all_labels = [], []

    def update(self, loss: float, preds: torch.Tensor, labels: torch.Tensor):
        self.epoch_loss += loss
        self.batch_count += 1
        self.all_preds.extend(F.softmax(preds, dim=1)[:, 1].detach().cpu().numpy())
        self.all_labels.extend(labels.detach().cpu().numpy())

    def get_epoch_metrics(self) -> Dict[str, float]:
        avg_loss = self.epoch_loss / max(1, self.batch_count)
        metrics = {'loss': avg_loss}
        if len(set(self.all_labels)) > 1:
            try:
                preds_binary = (np.array(self.all_preds) > 0.5).astype(int)
                metrics['auc'] = float(roc_auc_score(self.all_labels, self.all_preds))
                metrics['f1'] = float(f1_score(self.all_labels, preds_binary))
                metrics['precision'] = float(precision_score(self.all_labels, preds_binary))
                metrics['recall'] = float(recall_score(self.all_labels, preds_binary))
            except ValueError:
                metrics.update({'auc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0})
        return metrics


class EvolutionaryTrainer:
    def __init__(self, model, train_loader, val_loader, device, patience=5, lr=5e-5, weight_decay=0.01, max_lr=5e-5, epochs=50, grad_clip=1.0, amp=True, evolution_cfg=None, output_dir="outputs", wandb_cfg: Dict = None):
        self.model = model.to(device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.device = device
        self.criterion = FocalLoss(alpha=0.3, gamma=2.0)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        steps_per_epoch = max(1, len(train_loader))
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch)
        self.strategist = EvolutionaryStrategist(dim=model.dim, **(evolution_cfg or {}))
        self.patience = patience
        self.epochs_no_improve = 0
        self.best_val_auc = 0.0
        self.misclassified_samples: List[Tuple[Dict, Dict]] = []
        self.scaler = GradScaler(enabled=amp)
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.amp = amp
        self.output_dir = output_dir
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        self.log_path = os.path.join(self.output_dir, "logs", "train.log")
        # wandb
        self.wandb_cfg = wandb_cfg or {}
        self.use_wandb = bool(self.wandb_cfg.get("enabled", False) and wandb is not None)
        if self.use_wandb:
            wandb_mode = self.wandb_cfg.get("mode", None)
            if wandb_mode:
                os.environ["WANDB_MODE"] = wandb_mode
            wandb.init(project=self.wandb_cfg.get("project", "evohm"), name=self.wandb_cfg.get("run_name", None), config=self.wandb_cfg.get("config", {}))

    def _run_epoch(self, epoch: int, is_training: bool):
        self.model.train(is_training)
        loader = self.train_loader if is_training else self.val_loader
        metrics_tracker = EpochMetricsTracker()
        if not is_training:
            self.misclassified_samples = []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [{'Train' if is_training else 'Val'}]", leave=False, ncols=120)
        with torch.set_grad_enabled(is_training):
            for batch_data, targets, items in pbar:
                batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                labels = targets['classification'].to(self.device)
                with autocast(enabled=self.amp):
                    outputs = self.model(batch_data)
                    loss = self.criterion(outputs, labels)
                if is_training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                else:
                    preds = torch.argmax(outputs, dim=1)
                    misclassified_mask = preds != labels
                    if misclassified_mask.any():
                        misclassified_batch = {k: v[misclassified_mask] for k, v in batch_data.items()}
                        cpu_misclassified_mask = misclassified_mask.cpu()
                        misclassified_targets = {k: v[cpu_misclassified_mask] for k, v in targets.items()}
                        self.misclassified_samples.append((misclassified_batch, misclassified_targets))
                metrics_tracker.update(float(loss.item()), outputs, labels)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        return metrics_tracker.get_epoch_metrics()

    def _log_jsonl(self, record: Dict):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if self.use_wandb:
            flat = {}
            for scope in ("train", "val"):
                if scope in record and isinstance(record[scope], dict):
                    for k, v in record[scope].items():
                        flat[f"{scope}/{k}"] = v
            if "best_val_auc" in record:
                flat["val/best_auc"] = record["best_val_auc"]
            flat["epoch"] = record.get("epoch", 0)
            wandb.log(flat)

    def print_epoch_summary(self, epoch: int, train_metrics: Dict, val_metrics: Dict, evolution_events: List[Dict]):
        print(f"\n{'='*80}\nEPOCH {epoch+1} SUMMARY - Generation {self.model.generation}\n{'='*80}")
        print(f"Train: Loss={train_metrics['loss']:.4f} | AUC={train_metrics.get('auc', 0):.4f} | F1={train_metrics.get('f1', 0):.4f}")
        print(f"Valid: Loss={val_metrics['loss']:.4f} | AUC={val_metrics.get('auc', 0):.4f} | F1={val_metrics.get('f1', 0):.4f}")
        print(f"\nModules ({len(self.model.neural_modules)} active):")
        sorted_modules = sorted(self.model.neural_modules, key=lambda m: self.model.module_contributions.get(m.module_id, 0), reverse=True)
        for mod in sorted_modules:
            contrib = self.model.module_contributions.get(mod.module_id, 0.0)
            probation_mark = " (P)" if self.model.probation_info and self.model.probation_info['id'] == mod.module_id else ""
            config_str = ", ".join([f"{k}:{v}" for k, v in mod.config.items()])
            print(f"  id={mod.module_id} type={mod.module_type.upper()} contrib={contrib:.4f}{probation_mark} | {config_str}")
        if evolution_events:
            print("\nEvolution events:")
            for event in evolution_events:
                print(f"  {event}")
        print(f"\n{'='*80}")

    def train(self):
        main_pbar = tqdm(range(self.epochs), desc='Total Progress', ncols=100)
        for epoch in main_pbar:
            self.model.generation += 1
            train_metrics = self._run_epoch(epoch, is_training=True)
            val_metrics = self._run_epoch(epoch, is_training=False)
            evolution_events = self.strategist.manage_evolution_step(self.model, val_metrics.get('auc', 0.0), self.misclassified_samples)
            current_auc = val_metrics.get('auc', 0.0)
            if current_auc > self.best_val_auc:
                self.best_val_auc = current_auc
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, "checkpoints", "best_model.pth"))
                print(f"New best validation AUC: {self.best_val_auc:.4f}. Model saved.")
            else:
                self.epochs_no_improve += 1
            self.print_epoch_summary(epoch, train_metrics, val_metrics, evolution_events)
            record = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
                "best_val_auc": self.best_val_auc,
                "events": evolution_events,
            }
            self._log_jsonl(record)
            main_pbar.set_postfix({'Val_AUC': f"{current_auc:.4f}", 'Best_AUC': f"{self.best_val_auc:.4f}"})
            if self.epochs_no_improve >=  self.strategist.cooldown_epochs + 2:
                print("Early stopping triggered due to plateau.")
                break
        print(f"Training completed. Best validation AUC: {self.best_val_auc:.4f}")
        if self.use_wandb:
            wandb.finish()
        return self.model
