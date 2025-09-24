## Evo-Hateful-Memes: Evolutionary Cross-Modal Routing for Hateful Memes

An evolutionary cross-modal architecture that routes fused image–text features through a population of learnable modules (MLP / Transformer / ResNet). The system automatically adds, tests (probation), and prunes modules based on contribution to validation performance.

- **Data**: Facebook Hateful Memes
- **Backbones**: DistilBERT (text) + CLIP ViT-B/32 (image)
- **Fusion**: Cross-modal multi-head attention
- **Router**: Graph-attention over module outputs
- **Evolution**: Contribution-based add/prune with error-driven specialization

### TL;DR
```bash
# Windows PowerShell (venv recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Prepare dataset directory (see Dataset section)

# Single GPU
python scripts/train.py --config configs/default.yaml

# 2×A100 (80GB each) on one node
torchrun --nproc_per_node=2 scripts/train.py --config configs/ddp_2xa100.yaml

# Evaluate a checkpoint
python scripts/eval.py --config configs/default.yaml --ckpt outputs/checkpoints/best_model.pth
```

---

### 1. Abstract (ICLR-style)
We introduce an evolutionary routing framework for multimodal classification on Hateful Memes. Fused image-text features are dynamically routed through a population of diverse neural modules. The system adaptively adds and prunes modules using attention-derived contribution scores and misclassification analysis. This yields a flexible architecture that can specialize over time while avoiding over-parameterization.

### 2. Motivation and Intuition
Real-world multimodal problems contain heterogeneous phenomena (e.g., text sarcasm vs. visual cues). A single fixed pathway tends to overfit dominant patterns and underfit rare ones. Inspired by population-based learning and modular routing, we maintain a small, evolving set of pathways that compete/cooperate via attention. New modules are incubated under probation with a slight bias, and only survive if they contribute.

### 3. Contributions
- **Evolutionary cross-modal router** that learns when and how to use specialized modules.
- **Contribution-guided add/prune** mechanism with a short probation to reduce architectural churn.
- **Error-driven specialization**: new modules can be evolved from the best parent to target difficult examples.
- **Reproducible, modular codebase** with configs, single/Distributed training, and a minimal demo.

### 4. Method Summary
- Text features: DistilBERT [CLS], projected to `dim`.
- Image features: CLIP ViT-B/32 pooler output, projected to `dim`.
- Fusion: Cross-attn from image→text, residual + FFN.
- Module population: {Enhanced MLP, Small Transformer block, 1D ResNet-style block}. Each maps `dim→dim`.
- Router: Multi-head attention over module outputs, with a gating residual to fused features.
- Evolution:
  - Add: exploratory (new type) or exploitative (specialist from best parent).
  - Probation: biased attention; must exceed a contribution threshold in a few epochs.
  - Prune: zero-contribution modules for `longevity_threshold` epochs are removed.

### 5. Repository Layout
```
.
├─ src/
│  └─ evohm/
│     ├─ data.py           # Dataset & dataloaders
│     ├─ losses.py         # Focal loss
│     ├─ modules.py        # MLP/Transformer/ResNet blocks + NeuralModule
│     ├─ extractors.py     # DistilBERT & CLIP ViT feature extractors
│     ├─ fusion.py         # Cross-modal fusion
│     ├─ router.py         # GraphAttentionRouter
│     ├─ model.py          # EvolutionaryNeuralArchitecture
│     ├─ strategist.py     # EvolutionaryStrategist (add/prune/probation)
│     └─ trainer.py        # Training loop, metrics, logging
├─ scripts/
│  ├─ train.py             # CLI training (single & DDP via torchrun)
│  └─ eval.py              # Evaluation on dev split
├─ configs/
│  ├─ default.yaml         # Baseline config (matches notebook defaults)
│  └─ ddp_2xa100.yaml      # 2×A100 DDP settings
├─ hydra_train.py          # Optional Hydra-based entrypoint
├─ notebooks/
│  ├─ demo.py              # Minimal 1-epoch demo script
│  └─ .gitkeep
├─ Dockerfile              # CUDA-ready container
├─ requirements.txt        # Python dependencies
├─ pyproject.toml          # Packaging metadata
└─ README.md
```

### 6. Environment Setup
- Python ≥ 3.10
- Windows PowerShell example:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
- Optional: Docker + NVIDIA Container Toolkit
```bash
# Build
docker build -t evohm:latest .
# Run with 2 GPUs (mount project for outputs)
docker run --gpus all -it --rm -v %cd%:/workspace evohm:latest ^
  torchrun --nproc_per_node=2 scripts/train.py --config configs/ddp_2xa100.yaml
```

### 7. Dataset
Place the Hateful Memes dataset as:
```
project_root/
  data/
    hateful_memes/
      train.jsonl
      dev_seen.jsonl
      img/ (or images in the same folder referenced by jsonl)
```
- If your images live elsewhere, update `configs/default.yaml`:
```yaml
data:
  root: data/hateful_memes
  train_file: train.jsonl
  val_file: dev_seen.jsonl
  image_dir: .
```
Set `image_dir` accordingly (relative to `root`).

### 8. Experimental Setup (ICLR-style)
- Hardware: 2×NVIDIA A100 (80GB) single node; also runs on 1 GPU.
- Software: PyTorch 2.2, Transformers ≥4.42, CUDA 12.x runtime container (see Dockerfile), Windows instructions included.
- Training: AdamW, OneCycleLR, focal loss (α=0.3, γ=2.0), grad clip 1.0, AMP on.
- Batch size/sequence length: defined in `configs/default.yaml` (`data.batch_size`, `data.seq_len`).
- Initialization: Start with Transformer and MLP modules; evolution may add/prune further modules.

### 9. Reproduction Protocol
1) Install dependencies and prepare the dataset (Sections 6 and 7).
2) Verify GPUs are visible:
```bash
python -c "import torch; print(torch.cuda.device_count());"
```
3) Single GPU run (baseline):
```bash
python scripts/train.py --config configs/default.yaml
```
4) Multi-GPU (2×A100) run:
```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/ddp_2xa100.yaml
```
5) Evaluate best checkpoint:
```bash
python scripts/eval.py --config configs/default.yaml --ckpt outputs/checkpoints/best_model.pth
```
6) Enable W&B logging (optional):
```yaml
wandb:
  enabled: true
  project: evohm
  run_name: evohm-a100-2gpu
  mode: online
```
7) Hydra alternative (with overrides):
```bash
python hydra_train.py optim.epochs=10 data.batch_size=32 wandb.enabled=true
```

### 10. Expected Results
- Validation AUC typically > 0.60 after early epochs with evolution; exact scores vary with seed and dataset placement.
- Monitor module contributions and evolution events in logs.

### 11. Limitations
- Uses frozen backbones with partial unfreezing; full fine-tuning may improve results but increases compute.
- Attention-based contribution is a proxy and can be noisy early in training.
- Evolution adds stochasticity; results may vary across runs without strict determinism settings.

### 12. Broader Impact & Ethics
- Hateful content classification systems risk false positives/negatives. Use in-the-loop human moderation.
- Dataset biases can propagate; evaluate fairness across subgroups and consider bias mitigation strategies.
- Follow dataset license and usage restrictions; do not repurpose for surveillance.

### 13. Troubleshooting
- “No training data found”: verify dataset paths in `configs/default.yaml`.
- CUDA OOM: lower `data.batch_size` or `model.dim`.
- Transformers download issues: ensure first-run internet or pre-cache models.
- Windows dataloader slowness: set `data.num_workers` to 0–2.

### 14. Reproducibility Checklist
- Random seeds fixed (Python/Torch) where practical.
- Config files versioned under `configs/`.
- Checkpointing and separate evaluation script.
- Determinism caveats for CUDA/cuDNN and distributed execution.

### 15. License & Acknowledgements
- Backbones and datasets follow their respective licenses.
- Built with PyTorch and Hugging Face Transformers.

### 16. Citation
```text
@software{evo_hateful_memes,
  title  = {Evo-Hateful-Memes: Evolutionary Cross-Modal Routing for Hateful Memes},
  year   = {2025},
  author = {Anonymous},
  url    = {https://github.com/anonymous/evo-hateful-memes}
}
```
