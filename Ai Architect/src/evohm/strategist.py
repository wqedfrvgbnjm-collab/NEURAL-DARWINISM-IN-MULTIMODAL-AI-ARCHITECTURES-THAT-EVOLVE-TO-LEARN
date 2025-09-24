from collections import defaultdict
from typing import Dict, List, Tuple
import random
import torch

from .modules import ModuleType


class EvolutionaryStrategist:
    def __init__(self, longevity_threshold: int = 4, cooldown_epochs: int = 3, dim: int = 512):
        self.longevity_threshold = longevity_threshold
        self.cooldown_epochs = cooldown_epochs
        self.failed_additions = defaultdict(int)
        self.all_module_types = [ModuleType.MLP, ModuleType.TRANSFORMER, ModuleType.RESNET]
        self.dim = dim

    def manage_evolution_step(self, model, last_val_auc: float, misclassified_samples: List[Tuple[Dict, Dict]]):
        self.model = model
        events = []
        probation_ended_successfully = self._update_module_status()
        if probation_ended_successfully:
            events.append({'action': 'PROBATION_SUCCESS', 'module_type': self.model.probation_info['type'], 'reason': 'Module gained contribution and was integrated.'})
            self.model.probation_info = None
        prune_event = self._prune_dead_weight()
        if prune_event:
            events.append(prune_event)
        if len(self.model.neural_modules) < 6 and not self.model.probation_info:
            add_event = self._add_new_module(misclassified_samples)
            if add_event:
                events.append(add_event)
        for m_type in list(self.failed_additions.keys()):
            self.failed_additions[m_type] -= 1
            if self.failed_additions[m_type] <= 0:
                del self.failed_additions[m_type]
        return events

    def _update_module_status(self):
        if not self.model.probation_info:
            return False
        prob_id = self.model.probation_info['id']
        module_on_probation = next((m for m in self.model.neural_modules if m.module_id == prob_id), None)
        if module_on_probation:
            module_on_probation.epochs_existed += 1
            contribution = self.model.module_contributions.get(prob_id, 0.0)
            if contribution > 0.05:
                return True
            elif module_on_probation.epochs_existed >= self.longevity_threshold:
                self.model.remove_module_by_id(prob_id, f"Failed probation with contribution {contribution:.4f}")
                self.failed_additions[module_on_probation.module_type] = self.cooldown_epochs
                self.model.probation_info = None
        else:
            self.model.probation_info = None
        return False

    def _prune_dead_weight(self):
        for module in self.model.neural_modules:
            if self.model.probation_info and module.module_id == self.model.probation_info.get('id'):
                continue
            contribution = self.model.module_contributions.get(module.module_id, 0.0)
            if contribution < 0.01:
                module.epochs_with_zero_contribution += 1
            else:
                module.epochs_with_zero_contribution = 0
            if module.epochs_with_zero_contribution >= self.longevity_threshold and len(self.model.neural_modules) > 2:
                reason = f"Pruned due to zero contribution for {self.longevity_threshold} epochs."
                self.model.remove_module_by_id(module.module_id, reason)
                return {'action': 'REMOVE', 'module_type': module.module_type, 'reason': reason}
        return None

    @torch.no_grad()
    def _get_failure_analysis(self, misclassified_samples):
        if not misclassified_samples or not self.model.neural_modules:
            return None
        self.model.eval()
        error_contributions = defaultdict(lambda: {'sum': 0.0, 'count': 0})
        for batch_data, _ in misclassified_samples:
            device = next(self.model.parameters()).device
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            _, attn_weights = self.model(batch_data, return_attn=True)
            if attn_weights.numel() > 0:
                for i, module in enumerate(self.model.neural_modules):
                    if i < attn_weights.shape[1]:
                        error_contributions[module.module_type]['sum'] += attn_weights[:, i].sum().item()
                        error_contributions[module.module_type]['count'] += attn_weights.shape[0]
        avg_error_contrib = {t: v['sum'] / v['count'] for t, v in error_contributions.items() if v['count'] > 0}
        overall_contrib = defaultdict(lambda: {'sum': 0.0, 'count': 0})
        for m in self.model.neural_modules:
            contrib = self.model.module_contributions.get(m.module_id, 0.0)
            overall_contrib[m.module_type]['sum'] += contrib
            overall_contrib[m.module_type]['count'] += 1
        avg_overall_contrib = {t: v['sum'] / v['count'] for t, v in overall_contrib.items()}
        ratios = {t: avg_error_contrib.get(t, 0) / avg_overall_contrib.get(t, 1e-6) for t in avg_overall_contrib}
        if not ratios:
            return None
        worst_performing_type = min(ratios, key=ratios.get)
        if ratios[worst_performing_type] < 0.8:
            return worst_performing_type
        return None

    def _evolve_specialist_module(self, target_type: str):
        candidate_parents = [m for m in self.model.neural_modules if m.module_type == target_type and not (self.model.probation_info and m.module_id == self.model.probation_info.get('id'))]
        if not candidate_parents:
            return None, None
        parent = max(candidate_parents, key=lambda m: self.model.module_contributions.get(m.module_id, 0.0))
        new_config = parent.config.copy()
        if target_type == ModuleType.TRANSFORMER:
            current_nhead = new_config.get('nhead', 8)
            possible_nheads = [h for h in [4, 8, 16] if self.dim % h == 0 and h != current_nhead]
            if possible_nheads:
                new_config['nhead'] = random.choice(possible_nheads)
            new_config['dropout'] = max(0.1, min(0.5, new_config.get('dropout', 0.3) + random.uniform(-0.1, 0.1)))
        elif target_type == ModuleType.MLP:
            new_config['dropout_rate'] = max(0.1, min(0.5, new_config.get('dropout_rate', 0.4) + random.uniform(-0.15, 0.15)))
        elif target_type == ModuleType.RESNET:
            new_config['n_layers'] = max(1, min(4, new_config.get('n_layers', 2) + random.choice([-1, 1])))
        reason = f"Evolved {target_type.upper()} from module {parent.module_id} to fix performance gap."
        return new_config, reason

    def _add_new_module(self, misclassified_samples):
        target_type = self._get_failure_analysis(misclassified_samples)
        new_config, reason = None, None
        if target_type and random.random() < 0.75:
            new_config, reason = self._evolve_specialist_module(target_type)
        if not new_config:
            candidate_types = [t for t in self.all_module_types if t not in self.failed_additions]
            if not candidate_types:
                return None
            selected_type = random.choice(candidate_types)
            if selected_type == ModuleType.TRANSFORMER:
                new_config = {'nhead': 8, 'dropout': 0.3}
            elif selected_type == ModuleType.MLP:
                new_config = {'dropout_rate': 0.4}
            elif selected_type == ModuleType.RESNET:
                new_config = {'n_layers': 2, 'dropout_rate': 0.4}
            reason = f"Exploration: Adding foundational '{selected_type.upper()}'."
            target_type = selected_type
        new_module = self.model.add_module(target_type, reason, new_config)
        self.model.probation_info = {'id': new_module.module_id, 'weight': 0.25, 'type': new_module.module_type}
        return {'action': 'ADD', 'module_type': target_type, 'module_id': new_module.module_id, 'reason': reason}
