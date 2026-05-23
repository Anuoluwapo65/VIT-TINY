import torch     
import torch.nn as nn      
from checkpoint import checkpoint_path
import os       
import pytorch_lightning as pl    
from train import VIT_Transformer     
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from dataloader import train_dataloader, val_dataloader, test_dataloader

def train_model(**kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(checkpoint_path, "vitmodel"),
                         accelerator = "auto", 
                         devices = 1, 
                         max_epochs = 180,
                         min_epochs = 30,
                         callbacks = [ModelCheckpoint(save_weights_only = True, mode = "max",monitor = "val_acc"), LearningRateMonitor("epoch")])
    if trainer.logger is not None:
     trainer.logger._log_graph = True
     trainer.logger._default_hp_metrics = None




    pretrained_filename = os.path.join(checkpoint_path, "VIT.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Downloading this model from pretrained_files {pretrained_filename}")

        model = VIT_Transformer.load_from_checkpoint(pretrained_filename)

    else:

        pl.seed_everything(42)

        model = VIT_Transformer(**kwargs)
        trainer.fit(model, train_dataloader, val_dataloader)
        model = VIT_Transformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)


    val_result = trainer.validate(model, val_dataloader, verbose = False)
    test_result = trainer.test(model, test_dataloader, verbose = False)


    results = {"val_results" : val_result[0] ["val_acc"], "test_results": test_result[0] ["test_acc"]}

    return model,  results




if __name__ == "__main__":
   model, results = train_model(model_kwargs={
                                'embed_dim': 256,
                                'hidden_dim': 512,
                                'num_heads': 8,
                                'num_layers': 6,
                                'patch_size': 4,
                                'num_channels': 3,
                                'num_patches': 64,
                                'num_classes': 10,
                                'dropout': 0.2
                            },
                            lr=3e-4)
   print("ViT results", results)
   print(f"Validation accuracy: {results['val_results']:.4f}")
   print(f"Test accuracy: {results['test_results']:.4f}")









    







