# Tiny Vision Transformer in PyTorch

This project is a small Vision Transformer (ViT) implementation for CIFAR-10 image classification using PyTorch and PyTorch Lightning. It builds image patches, projects them into an embedding space, applies stacked transformer encoder blocks, and predicts classes from a learned CLS token.

The repository includes:

- A ViT encoder built from patch embeddings, positional encodings, and multi-head self-attention
- A PyTorch Lightning training module
- CIFAR-10 dataloaders with augmentation and normalization
- Checkpoint and TensorBoard logging support
- Saved checkpoints and logs for previous runs

## Model Summary

The default configuration in `model.py` uses:

- `embed_dim=256`
- `hidden_dim=512`
- `num_heads=8`
- `num_layers=6`
- `patch_size=4`
- `num_channels=3`
- `num_patches=64`
- `num_classes=10`
- `dropout=0.2`

The model pipeline is:

1. Split each `32x32` CIFAR-10 image into `4x4` patches
2. Flatten each patch and project it with a linear layer
3. Add a learned CLS token
4. Add learned positional encodings
5. Pass the sequence through stacked transformer blocks
6. Classify with the CLS token representation

## Dataset

This project trains on CIFAR-10 from `torchvision.datasets.CIFAR10`.

Training data uses:

- `RandomHorizontalFlip`
- `RandomVerticalFlip`
- `RandomResizedCrop`
- Channel normalization using dataset mean and standard deviation

Validation and test data use normalization only.

## Project Structure

- `model.py`: training entry point and evaluation
- `train.py`: PyTorch Lightning module
- `encoder.py`: Vision Transformer model
- `transformer.py`: transformer encoder block
- `dataloader.py`: CIFAR-10 loading, transforms, and patch extraction
- `checkpoint.py`: paths, seed setup, and pretrained file download
- `setup.py`: package metadata and dependencies

## Installation

Create an environment and install the required packages:

```bash
pip install torch torchvision numpy matplotlib pytorch-lightning wandb tensorboard
```

Or install the project package:

```bash
pip install -e .
```

## How To Run

Train or evaluate the model with:

```bash
python model.py
```

What happens when you run it:

- If a pretrained checkpoint exists at `checkpoint_path/VIT.ckpt`, it will be loaded
- Otherwise, the model will train and the best checkpoint will be restored
- The script then runs validation and test evaluation
- Final metrics are printed from the `results` dictionary

Example output:

```text
ViT results {'val_results': 0.6806, 'test_results': 0.6514}
Validation accuracy: 0.6806
Test accuracy: 0.6514
```

## Confirmed Results

From the saved local checkpoint in `checkpoint_path/vitmodel/lightning_logs/version_3/checkpoints/epoch=170-step=53523.ckpt`, the confirmed metrics are:

- Validation accuracy: `68.06%` final
- Best validation accuracy: `68.57%`
- Test accuracy: `65.14%`

The repository also contains an external tutorial log in `checkpoint_path/tutorial15/tensorboards/ViT` with a higher saved validation accuracy around `77.82%`, but that is separate from the local checkpoint above.

## Training Details

- Framework: PyTorch Lightning
- Optimizer: `Adam`
- Learning rate: `3e-4`
- LR scheduler: `MultiStepLR`
- LR milestones: `[100, 120]`
- Max epochs: `180`
- Min epochs: `30`
- Checkpoint monitor: `val_acc`

## Notes

- The dataloader prints CIFAR-10 channel mean and standard deviation when imported
- The current dataloaders use `num_workers=0`
- The repo includes saved event files and checkpoints under `checkpoint_path/`
- Some filenames and package references appear inconsistent, such as `transfomer` in `setup.py`; the current training code still works from the local source files

## Author

Abidoye Anuoluwapo
