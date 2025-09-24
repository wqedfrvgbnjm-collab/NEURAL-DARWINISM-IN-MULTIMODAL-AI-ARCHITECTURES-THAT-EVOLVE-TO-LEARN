## NEURAL DARWINISM IN MULTIMODAL AI: ARCHITECTURES THAT EVOLVE TO LEARN

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

### 1. Abstract 
We introduce an evolutionary routing framework for multimodal classification on Hateful Memes. Fused image-text features are dynamically routed through a population of diverse neural modules. The system adaptively adds and prunes modules using attention-derived contribution scores and misclassification analysis. This yields a flexible architecture that can specialize over time while avoiding over-parameterization.

### 2. Method Summary
- Text features: DistilBERT [CLS], projected to `dim`.
- Image features: CLIP ViT-B/32 pooler output, projected to `dim`.
- Fusion: Cross-attn from image→text, residual + FFN.
- Module population: {Enhanced MLP, Small Transformer block, 1D ResNet-style block}. Each maps `dim→dim`.
- Router: Multi-head attention over module outputs, with a gating residual to fused features.
- Evolution:
  - Add: either exploratory (new type) or exploitative (specialist evolved from best parent).
  - Probation: new module receives a bias; must reach a contribution threshold in a few epochs.
  - Prune: modules with near-zero contribution for `longevity_threshold` epochs are removed.

### 3. Repository Layout
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

### 4. Environment Setup
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

### 5. Dataset
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

### 6. Reproduction Instructions
- We used 2×A100 (80GB each). The code also works on a single GPU.
- Determinism: seeds set in Python/Torch; due to CUDA and dataloader, expect minor variance.

A) Single GPU (baseline)
```bash
python scripts/train.py --config configs/default.yaml
```

B) Two GPUs (same node)
```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/ddp_2xa100.yaml
```

C) Hydra (optional, override params in-line)
```bash
python hydra_train.py optim.epochs=10 data.batch_size=32 wandb.enabled=true
```

D) Evaluate a saved checkpoint
```bash
python scripts/eval.py --config configs/default.yaml --ckpt outputs/checkpoints/best_model.pth
```

E) Override common hyperparameters (without Hydra)
```bash
python scripts/train.py --config configs/default.yaml \
  --override optim.lr=3e-5 optim.epochs=25 data.batch_size=16 model.dim=512
```


### 7. Weights & Biases (optional)
Enable in `configs/default.yaml`:
```yaml
wandb:
  enabled: true
  project: evohm
  run_name: evohm-baseline
  mode: online   # or offline
```
Logged metrics include `train/*`, `val/*`, and `val/best_auc` per epoch.

### 8. Troubleshooting
- “No training data found”: verify dataset paths in `configs/default.yaml`.
- CUDA OOM: lower `data.batch_size` or reduce `model.dim`.
- CLIP/Transformers download issues: ensure internet access on first run or pre-download models to the Hugging Face cache.
- Slow dataloader on Windows: reduce `data.num_workers` to 0–2.

### 9. Reproducibility Checklist
- Random seeds fixed for Python/Torch.
- Config files versioned in `configs/`.
- Best checkpoint saved and evaluation script provided.
- Determinism caveat for CUDA/cuDNN and DDP.

### 10. License & Acknowledgements
- This code references ideas and backbones from the Hateful Memes benchmark and Hugging Face Transformers/CLIP.
- DistilBERT and CLIP are used under their respective licenses.

### 11. Citation
```text
@software{evo_hateful_memes,
  title  = {Evo-Hateful-Memes: Evolutionary Cross-Modal Routing for Hateful Memes},
  year   = {2025},
  author = {Anonymous},
  url    = {https://github.com/anonymous/evo-hateful-memes}
}
```
