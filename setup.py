from setuptools import setup

setup(
    name="tiny-vision-transformer-pytorch",
    version="0.1.0",
    author="Abidoye Anuoluwapo",
    description="A PyTorch implementation of the Tiny Vision Transformer architecture for image classification.",
    py_modules=[
        "checkpoint",
        "dataloader",
        "mlp",
        "model",
        "module",
        "train",
        "transfomer",
    ],
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "pytorch-lightning",
        "wandb",
        "tensorboard",
    ],
    python_requires=">=3.8",
)
