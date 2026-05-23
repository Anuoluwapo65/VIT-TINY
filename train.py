import torch     
import torch.nn as nn     
import pytorch_lightning as pl
from encoder import VIT_transformer
class VIT_Transformer(pl.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VIT_transformer(**model_kwargs)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100, 120], gamma = 0.1)
        return [optimizer], [scheduler]
    

    def _calculate_loss(self, batch, mode = "train"):
        label, target = batch
        pred = self.model(label)
        loss = self.loss(pred, target)
        acc = (pred.argmax(dim = -1) == target).float().mean()
        
        
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

         


    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode = "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss =  self._calculate_loss(batch, mode = "val")
        return loss
        
    
    def test_step(self, batch, batch_idx):
        loss =  self._calculate_loss(batch, mode = "test")
        return loss
        
